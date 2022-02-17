from scipy.stats import multivariate_normal
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
import torch
from tools.video_readers import TorchIterableFromReader
from time import time
from detection.transforms import TransformFrames
from collections import defaultdict
from moviepy.editor import ImageSequenceClip
from skimage.transform import downscale_local_mean

class GaussianMixture(object):
    def __init__(self, means, covariance, weights):
        self.components = [multivariate_normal(
            mean=mean, cov=covariance) for mean in means]
        self.weights = weights

    def pdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.pdf(x)
        return result

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.cdf(x)
        return result

def init_trackers(engine, detections, confs, labels, frame_nb, state_variance, observation_variance, delta):
    trackers = []

    for detection, conf, label in zip(detections, confs, labels):
        tracker_for_detection = engine(frame_nb, detection, conf, label, state_variance, observation_variance, delta)
        trackers.append(tracker_for_detection)

    return trackers

def exp_and_normalise(lw):
    w = np.exp(lw - lw.max())
    return w / w.sum()

def in_frame(position, shape, border=0.02):


    shape_x = shape[1]
    shape_y = shape[0]
    x = position[0]
    y = position[1]

    return x > border*shape_x and x < (1-border)*shape_x and y > border*shape_y and y < (1-border)*shape_y

def gather_filenames_for_video_in_annotations(video, images, data_dir):
    images_for_video = [image for image in images
                        if image['video_id'] == video['id']]
    images_for_video = sorted(
        images_for_video, key=lambda image: image['frame_id'])

    return [os.path.join(data_dir, image['file_name'])
                 for image in images_for_video]

def get_detections_for_video(reader, detector, batch_size=16, device=None):

    detections = []
    dataset = TorchIterableFromReader(reader, TransformFrames())
    loader = DataLoader(dataset, batch_size=batch_size)
    average_times = []
    with torch.no_grad():
        for preprocessed_frames in loader:
            time0 = time()
            detections_for_frames = detector(preprocessed_frames.to(device))
            average_times.append(time() - time0)
            for detections_for_frame in detections_for_frames:
                if len(detections_for_frame): detections.append(detections_for_frame)
                else: detections.append(np.array([]))
    print(f'Frame-wise inference time: {batch_size/np.mean(average_times)} fps')
    return detections


def generate_video_with_annotations(video, output_detected, output_filename, skip_frames, maxframes, downscale, logger):
    fps = 24
    logger.info("---intepreting json")
    results = defaultdict(list)
    for trash in output_detected["detected_trash"]:
        for k, v in trash["frame_to_box"].items():
            frame_nb = int(k) - 1
            object_nb = trash["id"] + 1
            object_class = trash["label"]
            center_x = v[0]
            center_y = v[1]
            results[frame_nb * (skip_frames+1)].append((object_nb, center_x, center_y, object_class))
            # append next skip_frames
            if str(frame_nb + 2) in trash["frame_to_box"]:
                next_trash = trash["frame_to_box"][str(frame_nb + 2)]
                next_x = next_trash[0]
                next_y = next_trash[1]
                for i in range(1, skip_frames+1):
                    new_x = center_x + (next_x - center_x) * i/(skip_frames+1)
                    new_y = center_y + (next_y - center_y) * i/(skip_frames+1)
                    results[frame_nb * (skip_frames+1) + i].append((object_nb, new_x, new_y, object_class))
    logger.info("---writing video")

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # writer = cv2.VideoWriter(filename=output_filename,
                                    #apiPreference=cv2.CAP_FFMPEG,
    #                                fourcc=fourcc,
    #                                fps=fps,
    #                                frameSize=video.shape)

    font = cv2.FONT_HERSHEY_COMPLEX
    ret, frame, frame_nb = video.read()
    frames = []
    while ret:
        detections_for_frame = results[frame_nb]
        for detection in detections_for_frame:
            cv2.putText(frame, f'{detection[0]}/{detection[3]}', (int(detection[1]), int(detection[2])+5), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        frame = downscale_local_mean(frame, (downscale,downscale,1)).astype(np.uint8)
        frames.append(frame[:,:,::-1])

        ret, frame, frame_nb = video.read()
        if frame_nb > maxframes:
            break

    clip = ImageSequenceClip(sequence=frames, fps=fps)
    clip.write_videofile(output_filename, fps=fps)
    del frames

    logger.info("---finished writing video")


def resize_external_detections(detections, ratio):

    for detection_nb in range(len(detections)):
        detection = detections[detection_nb]
        if len(detection):
            detection = np.array(detection)[:,:-1]
            detection[:,0] = (detection[:,0] + detection[:,2])/2
            detection[:,1] = (detection[:,1] + detection[:,3])/2
            detections[detection_nb] = detection[:,:2]/ratio
    return detections


def write_tracking_results_to_file(results, ratio_x, ratio_y, output_filename):
    """ writes the output result of a tracking the following format:
    - frame
    - id
    - x_tl, y_tl, w=0, h=0
    - 4x unused=-1
    """
    with open(output_filename, 'w') as output_file:
        for result in results:
            output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0]+1,
                                                                result[1]+1,
                                                                ratio_x * result[2],
                                                                ratio_y * result[3],
                                                                0, #width
                                                                0, #height
                                                                result[4],result[5],-1,-1))


def read_tracking_results(input_file):
    """ read the input filename and interpret it as tracklets
    i.e. lists of lists
    """
    raw_results = np.loadtxt(input_file, delimiter=',')
    if raw_results.ndim == 1: raw_results = np.expand_dims(raw_results,axis=0)
    tracklets = defaultdict(list)
    for result in raw_results:
        frame_id = int(result[0])
        track_id = int(result[1])
        left, top, width, height = result[2:6]
        center_x = left + width/2
        center_y = top + height/2
        conf = result[6]
        class_id = result[7]
        tracklets[track_id].append((frame_id, center_x, center_y, conf, class_id))

    tracklets = list(tracklets.values())
    return tracklets

def gather_tracklets(tracklist):
    """ Converts a list of flat tracklets into a list of lists
    """
    tracklets = defaultdict(list)
    for track in tracklist:
        frame_id = track[0]
        track_id = track[1]
        center_x = track[2]
        center_y = track[3]
        tracklets[track_id].append((frame_id, center_x, center_y))

    tracklets = list(tracklets.values())
    return tracklets

class FramesWithInfo:
    def __init__(self, frames, output_shape=None):
        self.frames = frames
        if output_shape is None:
            self.output_shape = frames[0].shape[:-1][::-1]
        else: self.output_shape = output_shape
        self.end = len(frames)
        self.read_head = 0

    def __next__(self):
        if self.read_head < self.end:
            frame = self.frames[self.read_head]
            self.read_head+=1
            return frame

        else:
            raise StopIteration

    def __iter__(self):
        return self
