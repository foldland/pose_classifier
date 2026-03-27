import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;

import 'ema_smoothing.dart';
import 'pose_classification_result.dart';
import 'pose_classifier.dart';
import 'pose_sample.dart';
import 'repetition_counter.dart';

/// A processor that classifies incoming poses.
class PoseClassifierProcessor {
  /// Creates an instance of [PoseClassifierProcessor].
  ///
  /// The [poseSamples] parameter is required and should contain the
  /// samples used for pose classification.
  ///
  /// Use [StreamingClassifierProcessor] for EMA smoothing and repetition
  /// counting.
  PoseClassifierProcessor({required List<PoseSample> poseSamples})
    : _poseClassifier = PoseClassifier(poseSamples);

  final PoseClassifier _poseClassifier;

  /// Given a new [mlkit.Pose] input, returns the classified pose.
  PoseClassificationResult classifyPose(mlkit.Pose pose) {
    final classification = _poseClassifier.classifyFromPose(pose);

    // Add maxConfidence class of current frame to result if pose is found.
    final maxConfidenceClass = classification.getMaxConfidenceClass();
    final maxConfidence =
        classification.getClassConfidence(maxConfidenceClass) /
        _poseClassifier.confidenceRange();

    return PoseClassificationResult(
      className: maxConfidenceClass,
      confidence: maxConfidence,
    );
  }
}

/// A processor that loads pose samples and classifies incoming poses.
///
/// This class is designed to handle pose classification and repetition
/// counting. It is important to call [classifyPose] off the UI thread
/// if heavy processing is involved, as Flutter isolates apply.
///
/// Accepts pose from a stream of [mlkit.Pose] for classification and repetition
/// counting.
class StreamingClassifierProcessor extends PoseClassifierProcessor {
  /// Creates a streaming instance of [StreamingClassifierProcessor].
  ///
  /// The [poseSamples] parameter is required and should contain the
  /// samples used for pose classification.
  ///
  ///
  /// This instance enables EMA smoothing and repetition counting.
  /// Specify [className] to

  StreamingClassifierProcessor({required super.poseSamples, String? className})
    : _emaSmoothing = EMASmoothing(),
      _repCounters = _repCounterFromSamples(poseSamples),
      _className = className;

  final EMASmoothing _emaSmoothing;
  final Map<String, RepetitionCounter> _repCounters;
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
  ({String name, int reps})? _repetitionCounter;

  /// Given a new [mlkit.Pose] input, returns the classified pose.
  ///
  /// Returns the repetition class with the highest wrap count.
  /// Specify poseType in the constructor to only track one type.
  @override
  StreamingClassificationResult classifyPose(mlkit.Pose pose) {
    // Return early without updating repCounter if no pose found.
    if (pose.landmarks.isEmpty) {
      return StreamingClassificationResult(
        repetitionClass: _repetitionCounter?.name,
        repetitionCount: _repetitionCounter?.reps,
      );
    }

    var classification = _poseClassifier.classifyFromPose(pose);

    // Update repetitionCounters if in stream mode.
    // Feed pose to smoothing even if no pose found.
    classification = _emaSmoothing.getSmoothedResult(classification);
    if (_className != null) {
      final repCounter = _repCounters[_className];
      final repsAfter =
          repCounter?.addClassificationResult(classification) ?? 0;

      _repetitionCounter = (name: _className, reps: repsAfter);
    } else {
      for (final repCounter in _repCounters.values) {
        final repsBefore = repCounter.numRepeats;
        final repsAfter = repCounter.addClassificationResult(classification);

        // First pose we find a repetition for must be the right one.
        if (repsAfter > repsBefore) {
          _repetitionCounter = (name: repCounter.className, reps: repsAfter);
          break;
        }
      }
    }

    // Add maxConfidence class of current frame to result if pose is found.
    final maxConfidenceClass = classification.getMaxConfidenceClass();
    final maxConfidence =
        classification.getClassConfidence(maxConfidenceClass) /
        _poseClassifier.confidenceRange();

    return StreamingClassificationResult(
      repetitionClass: _repetitionCounter?.name,
      repetitionCount: _repetitionCounter?.reps,

      className: maxConfidenceClass,
      confidence: maxConfidence,
    );
  }
}
