# NLF Pipeline For Multi-Person 3D Pose and Shape Estimation in Video

This repository contains tooling infrastructure to apply the model from the NeurIPS 2024 paper [Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation](https://github.com/isarandi/nlf) on videos. The NLF model itself is a single-person single-crop model at its core. To apply it to videos, we do the following:

- **Person tracking**: we use video object segmentation, namely a modified version of STCN
- **Camera motion tracking**: we use DROID-SLAM, acting on non-person pixels
- **Estimate 3D human pose and shape**: we use the NLF model on the bounding boxes of the STCN masks.
- **Temporal smoothing**: We apply a median-like (non-learned) temporal smoothing to the world-space vertex and joint predictions of NLF. 
- **Fitting SMPL**: we use the [SMPLfitter](https://github.com/isarandi/smplfitter) algorithm to fit the SMPL body model to the smoothed nonparametric vertex and joint trajectories. Body shape fitting is done on the level of an entire track, not per frame, for better consistency, and a matching scaling factor is estimated per frame to account for scale ambiguity.
- **Ground plane estimation**: we estimate the ground plane based on the distribution of foot locations throughout the video, using RANSAC and other steps.
- **Visualization trajectory generation**: based on the estimated camera trajectory and the human motion, we create a smooth visualization camera trajectory that shows the human from a side view, keeping the subject approximately centered in the frame.
- **Frame interpolation**: Using RIFE, we may double the framerate for visualization purposes, and also interpolate intermediate human poses. Visualization of human meshes at 24-30 fps often looks jarringly choppy, while 50-60 fps feel much smoother.
- **Rendering**: we use [PoseViz](https://github.com/isarandi/poseviz) or [BlendiPose](https://github.com/isarandi/blendipose) to render the results.

Setting up all these steps with the dependencies is quite involved. More extensive docs are coming soon. Below is an initial best effort to document the process.

## Installation

```bash
pip install git+https://github.com/isarandi/nlf-pipeline.git
```

## Initial setup and data

Choose a working directory and set `INFERENCE_ROOT`:

```bash
export INFERENCE_ROOT=mypath
```

Create the directory structure:

```bash
mkdir -p $INFERENCE_ROOT/{videos_in,videos_out,masks,masks_semseg,cameras,cameras_traj,preds,preds_np,smooth_fits}
```

Put the input video(s) in the input folder, as `$INFERENCE_ROOT/videos_in/${vid}.mp4`. Whatever is the 
name of the file before the extension will be the identifier of the video, also used for intermediate and result outputs.

## Single-command processing

To process a video from start to finish, you can use `run_all.py`. The simplest usage is:

```bash
python -m nlf_pipeline.run_all --video-id=$vid
```

A more elaborate example:

```bash
python -m nlf_pipeline.run_all --video-id=$vid --time-range=00:45-01:25 --seg-init-time=00:46 --max-persons=4 --viz --gui-selection
```

By specifying the `--time-range` option, the script will first cut out that segment of the video and process that part only. The `--seg-init-time` is the timestamp where the video object segmentation will be initialized. This is useful when the first frame of the video does not show all subjects clearly. The `--max-persons` option limits the number of people to track.
If you specify `--gui-selection`, then a window will open, showing the `--seg-init-time` moment of the video where you can click the subjects you want to track, then close the window.
The `--viz` option will visualize each step of the processing.

You can also specify `--static-camera`, then the camera motion estimation step will be skipped and the results will be relative to the camera's reference frame.

If a video with path `$INFERENCE_ROOT/videos_in/${vid}.mp4` does not exist, then the script will try to download it from YouTube, using the value of `--video-id` as the YouTube video ID.

## Step-by-step processing

### Step 1 [optional]: Video object segmentation (STCN)

To have coherent person tracks, it is beneficial to run a video object segmentation model first.

This is optional, and tracking can also be done later on the 3D estimates. The benefit of doing it at the beginning is that the later processing can be restricted to the target subjects, which saves quite some compute in crowded scenes where you only want to estimate a few people. Also, the tracks tend to be more coherent with VOS.

```bash
python -m stcnbuf.segment_video \
    --video-path="$INFERENCE_ROOT/videos_in/${vid}.mp4" \
    --output="$INFERENCE_ROOT/masks/${vid}_masks.pkl" \
    --init-frame=100 \
    --max-persons=2 \
    --permute-frames \
    --passes=2 \
    --morph-cleanup
```

This will segment an initial frame using Mask R-CNN, and then track the people using a modified STCN VOS model.
The `--init-frame` is where the Mask R-CNN initialization happens and this frame should clearly show all subjects. The highest-confidence detections here will be the ones who are tracked, at most `--max-persons` of them. Sometimes this "highest confidence" heuristic does not pick the subjects we actually care about. In this case, use the `--gui-selection` option, which will open a GUI where you can click which Mask R-CNN-detected instances from frame `init-frame` to track.

The `--permute-frames` shuffles the video frames before running segmentation. This helps STCN build a more diverse memory bank, which is beneficial for long videos where people's appearance may change, e.g. different zoom level, body orientation, clothes etc.

The `--passes` option allows to runs the segmentation multiple times, each time with a richer memory bank. Otherwise, the initial frames of the result may not be as good as the later ones. Intuitively,
once we've processed the full video and built up good representations of the subjects, it will be easier to segment the initial frames again.

The `--morph-cleanup` option applies some morphological cleanup operations to the masks, removing speckles, filling holes, removing mask parts that are far from the largest connected component etc.
This makes the bounding box of the resulting mask more reliable for the next steps.

### Step 2 [optional]: Camera motion estimation (DROID-SLAM)

As [Wang et al. (2024)](https://arxiv.org/abs/2403.17346) have shown, 3D human estimation with dynamic cameras can be tackled by estimating camera motion via SLAM, specifically DROID-SLAM.

This is an optional step. It can be skipped, and then all results will be relative to the camera's reference frame.

First, follow the instructions in the [Masked DROID-SLAM](https://github.com/isarandi/masked-droidslam) repository to install it.

Before running DROID-SLAM, we need to obtain semantic masks for the video. We might use the VOS masks from the previous step, but these may not cover all people in the scene, just the target subjects.
Instead, a fast way to get semantic masks is DeepLabV3, as follows:

```bash
python -m nlf_pipeline.run_semseg \
    --video-dir="$INFERENCE_ROOT/videos_in" \
    --file-pattern="${vid}.mp4" \
    --output-dir="$INFERENCE_ROOT/masks_semseg" \
    --mask-threshold=0.2 \
    --no-skip-existing \
    --batch-size=16
```

This saves the runlength-encoded binary masks ([rlemasklib](https://github.com/isarandi/rlemasklib)) of the person class into a pickle file under `$INFERENCE_ROOT/masks_semseg`.

Now we can estimate the camera motion:

```bash
python -m nlf_pipeline.run_slam \
    --droid-model-path="$DATA_ROOT/models/droid.pth" \
    --video-path="$INFERENCE_ROOT/videos_in/${vid}.mp4" \
    --mask-path="$INFERENCE_ROOT/masks_semseg/${vid}_masks.pkl" \
    --output-path="$INFERENCE_ROOT/cameras/${vid}.pkl" \
    --smooth
```

This saves the camera motion as a pickled list of [cameravision](https://github.com/isarandi/cameravision) `Camera` objects.

To verify that this step gave reasonable results, you can visualize the camera trajectory:

```bash
python -m nlf_pipeline.viz_camtraj --video-id=$vid --camera-view
```

### Step 3: Nonparametric 3D human pose and shape estimation (NLF)

Let's run the main part of the pipeline, the actual 3D human estimation. 

```bash
python -m nlf_pipeline.run_nlf \
    --model-path="https://bit.ly/nlf_l" \
    --video-dir="$INFERENCE_ROOT/videos_in" \
    --file-pattern="${vid}.mp4" \
    --output-dir="$INFERENCE_ROOT/preds_np" \
    --mask-dir="$INFERENCE_ROOT/masks" \
    --output-video-dir="$INFERENCE_ROOT/videos_out" \
    --camera-dir="$INFERENCE_ROOT/cameras" \
    --viz-downscale=2 \
    --batch-size=8 \
    --num-aug=1 \
    --internal-batch-size=8 \
    --antialias-factor=2 \
    --suppress-implausible-poses \
    --max-detections=-1 \
    --detector-threshold=0.15 \
    --detector-nms-iou-threshold=0.8 \
    --detector-flip-aug \
    --write-video \
    --viz   
```

This will save all the predicted vertex and joint coordinates for every frame and every subject in the video into a compressed format (using [BodyCompress](https://github.com/isarandi/bodycompress)) under `$INFERENCE_ROOT/preds_np` (np stands for nonparametric).

The `--viz` option will open a GUI window where you can see the 3D predictions in real time, visualized via [PoseViz](https://github.com/isarandi/poseviz).

### Step 4: SMPL fitting and temporal smoothing

The above results are already usable, but perform some post-processing. First, we can fit the SMPL model to the predicted vertices and joints. This will give us a more compact representation and allows us to leverage the body shape prior of SMPL. Also, many applications require SMPL parameters as input.

We can also apply temporal smoothing. Since the nonparametric predictions are frame-by-frame, there will be some jitter in the results.

Fitting and smoothing are intertwined because by fitting one shape vector to the whole track of each person, we get a more consistent output.

```bash
python -m nlf_pipeline.run_smoothed_smplfitter --video-id=$vid
```

This also estimates the ground plane based on the distribution of foot locations throughout the video, using RANSAC and other steps.

### Step 5: Rendering

We can now render the results with [PoseViz](https://github.com/isarandi/poseviz) or [BlendiPose](https://github.com/isaran/blendipose).


```bash
python -m nlf_pipeline.render --video-id=$vid --resolution 1920 1080 --viz
```
