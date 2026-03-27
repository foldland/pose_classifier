import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;

import 'ema_smoothing.dart';
import 'pose_classification_result.dart';
import 'pose_classifier.dart';
import 'pose_sample.dart';
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
  /// samples used for pose classification.
  ///
  /// Use [PoseClassifierProcessor.streaming] for EMA smoothing and repetition
  /// counting.
  PoseClassifierProcessor({required List<PoseSample> poseSamples})
    : _poseClassifier = PoseClassifier(poseSamples),
      _isStreamMode = false,
      _emaSmoothing = null,
      _repCounters = null,
      _className = null;

  /// Creates a streaming instance of [PoseClassifierProcessor].
  ///
  /// The [poseSamples] parameter is required and should contain the
  /// samples used for pose classification.
  ///
  ///
  /// This instance enables EMA smoothing and repetition counting.
  /// Specify [className] to
  PoseClassifierProcessor.streaming({
    required List<PoseSample> poseSamples,
    String? className,
  }) : _isStreamMode = true,
       _poseClassifier = PoseClassifier(poseSamples),
       _emaSmoothing = EMASmoothing(),
       _repCounters = _repCounterFromSamples(poseSamples),
       _className = className;

  final bool _isStreamMode;
  final EMASmoothing? _emaSmoothing;
  final Map<String, RepetitionCounter>? _repCounters;
  final PoseClassifier _poseClassifier;
  final String? _className;

  /// Creates repetition counters for all available pose types.
  ///
  /// Returns a Map of [RepetitionCounter] objects, one for each pose type in the [poseSamples],
  /// initialized with their corresponding class names.
  static Map<String, RepetitionCounter> _repCounterFromSamples(
    List<PoseSample> poseSamples,
  ) {
    final map = <String, RepetitionCounter>{};

    for (final PoseSample(:className) in poseSamples) {
      map[className] = RepetitionCounter(className);
    }

    return map;
  }

  // Cache last result if the current pose is empty.
  Repetition? _repetitionCounters;

  /// Given a new [mlkit.Pose] input, returns a Map of formatted String
  /// with pose classification results.
  ///
  /// The result contains up to two strings:
  /// 0: PoseClass : X reps
  /// 1: PoseClass : [0.0-1.0] confidence
  ///
  /// Returns the repetition class with the highest wrap count.
  /// Specify poseType in the constructor to only track one type.
  PoseClassificationResult? getPoseResult(mlkit.Pose pose) {
    // Return early without updating repCounter if no pose found.
    if (pose.landmarks.isEmpty) {
      return PoseClassificationResult(
        repetitions: _repetitionCounters,
        lastPose: null,
      );
    }

    var classification = _poseClassifier.classifyFromPose(pose);

    // Update repetitionCounters if in stream mode.
    if (_isStreamMode) {
      // Feed pose to smoothing even if no pose found.
      classification = _emaSmoothing!.getSmoothedResult(classification);

      if (_className == null) {
        for (final repCounter in _repCounters!.values) {
          final repsBefore = repCounter.numRepeats;
          final repsAfter = repCounter.addClassificationResult(classification);

          // First pose we find a repetition for must be the right one.
          if (repsAfter > repsBefore) {
            _repetitionCounters = (name: repCounter.className, reps: repsAfter);
            break;
          }
        }
      } else {
        final repCounter = _repCounters![_className];
        final repsAfter =
            repCounter?.addClassificationResult(classification) ?? 0;

        _repetitionCounters = (name: _className, reps: repsAfter);
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
