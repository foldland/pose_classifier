import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;

import 'ema_smoothing.dart';
import 'pose_classification_result.dart';
import 'pose_classifier.dart';
import 'pose_sample.dart';
import 'pose_type.dart';
import 'repetition_counter.dart';

/// A processor that loads pose samples and classifies incoming poses.
///
/// This class is designed to handle pose classification and repetition
/// counting. It is important to call `getPoseResult` off the UI thread
/// if heavy processing is involved, as Flutter isolates apply.
///
/// Accepts a stream of [mlkit.Pose] for classification and repetition
/// counting.
class PoseClassifierProcessor {
  /// Creates an instance of [PoseClassifierProcessor].
  ///
  /// The [poseSamples] parameter is required and should contain the
  /// samples used for pose classification. The [isStreamMode] parameter
  /// determines whether the processor operates in stream mode, which
  /// enables EMA smoothing and repetition counting.
  PoseClassifierProcessor({
    required List<PoseSample> poseSamples,
    bool isStreamMode = true,
  }) : _poseClassifier = PoseClassifier(poseSamples),
       _emaSmoothing = isStreamMode ? EMASmoothing() : null,
       _repCounters = isStreamMode ? PoseType.repCounters : null;

  final EMASmoothing? _emaSmoothing;
  final List<RepetitionCounter>? _repCounters;
  final PoseClassifier _poseClassifier;

  // Cahce last result if the current pose is empty.
  Repetition? _repetitionCounters;

  /// Given a new [mlkit.Pose] input, returns a Map of formatted String
  /// with pose classification results.
  ///
  /// The result contains up to two strings:
  /// 0: PoseClass : X reps
  /// 1: PoseClass : [0.0-1.0] confidence
  PoseClassificationResult? getPoseResult(mlkit.Pose pose) {
    // Return early without updating repCounter if no pose found.
    if (pose.landmarks.isEmpty) {
      return PoseClassificationResult(
        repetitions: _repetitionCounters,
        lastPose: null,
      );
    }

    var classification = _poseClassifier.classifyFromPose(pose);
    final isStreamMode = _repCounters != null && _emaSmoothing != null;

    // Update repetitionCounters if in stream mode.
    if (isStreamMode) {
      // Feed pose to smoothing even if no pose found.
      classification = _emaSmoothing.getSmoothedResult(classification);

      for (final repCounter in _repCounters) {
        final repsBefore = repCounter.numRepeats;
        final repsAfter = repCounter.addClassificationResult(classification);

        // First pose we find a repetition for must be the right one.
        if (repsAfter > repsBefore) {
          // Play a fun beep when rep counter updates.
          _repetitionCounters = (name: repCounter.className, reps: repsAfter);
          break;
        }
      }
    }

    // Add maxConfidence class of current frame to result if pose is found.
    final maxConfidenceClass = classification.getMaxConfidenceClass();
    final maxConfidence =
        classification.getClassConfidence(maxConfidenceClass) /
        _poseClassifier.confidenceRange();

    return PoseClassificationResult(
      repetitions: _repetitionCounters,
      lastPose: (name: maxConfidenceClass, confidence: maxConfidence),
    );
  }
}
