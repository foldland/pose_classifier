import 'classification_result.dart';
import 'pose_classifier.dart';

/// A counter that tracks repetitions of a specific pose class.
///
/// Uses confidence thresholds to determine when a pose enters and exits the
/// counted state. Works in conjunction with [PoseClassifier] to track exercise
/// repetitions.
///
/// Example:
/// ```dart
/// final counter = RepetitionCounter('squat');
/// final reps = counter.addClassificationResult(classificationResult);
/// ```
class RepetitionCounter {
  /// Creates a counter for tracking repetitions of [className].
  ///
  /// The [enterThreshold] defines the confidence level required to consider
  /// a pose as entered (default: 6.0).
  ///
  /// The [exitThreshold] defines the confidence level below which the pose
  /// is considered exited, triggering a rep count (default: 4.0).
  ///
  /// Both thresholds work with the Top K values from [PoseClassifier]
  /// which defaults to 10, making the valid range 0-10.
  RepetitionCounter(
    this.className, {
    double enterThreshold = 6,
    double exitThreshold = 4,
  }) : _enterThreshold = enterThreshold,
       _exitThreshold = exitThreshold;

  /// The name of the pose being counted.
  final String className;

  /// The confidence threshold required to enter the pose state.
  final double _enterThreshold;

  /// The confidence threshold below which the pose is considered exited.
  final double _exitThreshold;

  /// The current count of completed repetitions.
  int _numRepeats = 0;

  /// Returns the current number of completed repetitions.
  int get numRepeats => _numRepeats;

  /// Tracks whether the pose is currently in the entered state.
  bool _poseEntered = false;

  /// Processes a new pose classification result and updates the rep count.
  ///
  /// Takes a [classificationResult] containing confidence values for various
  /// pose classes and updates the repetition count if the configured pose
  /// transitions from entered to exited state.
  ///
  /// Returns the current total number of repetitions counted.
  int addClassificationResult(ClassificationResult classificationResult) {
    final poseConfidence = classificationResult.getClassConfidence(className);

    if (!_poseEntered) {
      _poseEntered = poseConfidence > _enterThreshold;
      return _numRepeats;
    }

    if (poseConfidence < _exitThreshold) {
      _numRepeats++;
      _poseEntered = false;
    }

    return _numRepeats;
  }

  @override
  String toString() {
    return 'RepetitionCounter(className: $className, numRepeats: $_numRepeats, _enterThreshold: $_enterThreshold, _exitThreshold: $_exitThreshold, _poseEntered: $_poseEntered)';
  }
}
