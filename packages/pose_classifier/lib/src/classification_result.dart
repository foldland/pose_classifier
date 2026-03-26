import 'pose_classifier.dart';

/// Represents Pose classification result as outputted by [PoseClassifier].
///
/// This class maintains a mapping of pose class names to their confidence
/// scores, which can be manipulated through various methods. The confidence
/// scores are based on how frequently a class appears in the top K nearest
/// neighbors.
class ClassificationResult {
  /// Creates a new empty classification result.
  ClassificationResult();

  /// Stores confidence scores for each pose class.
  ///
  /// The key is the class name, and the value is the confidence score.
  /// Confidence scores are in range [0, K] and can be float values after EMA
  /// smoothing.
  ///
  /// Higher values indicate greater confidence in the classification.
  final Map<String, double> _classConfidences = {};

  /// Returns a set of all pose names in this classification result.
  ///
  /// Returns an empty set if no poses are present.
  Set<String> getAllClasses() {
    return _classConfidences.keys.toSet();
  }

  /// Returns the confidence score for the specified pose class.
  ///
  /// If the class is not present in the classification result, returns 0.0.
  ///
  /// [className] The name of the pose class to look up.
  double getClassConfidence(String className) {
    return _classConfidences[className] ?? 0.0;
  }

  /// Returns the class name with the highest confidence score.
  ///
  /// If there are no classes in the classification result, returns an empty
  /// string. If multiple classes have the same highest confidence, returns
  /// the first one encountered.
  String getMaxConfidenceClass() {
    if (_classConfidences.isEmpty) {
      return '';
    }

    return _classConfidences.entries
        .reduce((a, b) => a.value >= b.value ? a : b)
        .key;
  }

  /// Increments the confidence score for the specified class by 1.0.
  ///
  /// If the class doesn't exist, it will be created with an initial score of
  /// 1.0.
  ///
  /// [className] The name of the pose class to increment.
  void incrementClassConfidence(String className) {
    _classConfidences[className] = getClassConfidence(className) + 1.0;
  }

  /// Sets the confidence score for the specified class to the given value.
  ///
  /// [className] The name of the pose class to update.
  /// [confidence] The new confidence score to set.
  void setClassConfidence(String className, double confidence) {
    _classConfidences[className] = confidence;
  }

  @override
  String toString() {
    final entries = _classConfidences.entries
        .map((e) => '  ${e.key}: ${e.value.toStringAsFixed(2)}')
        .join(',\n');
    return 'ClassificationResult(\n$entries\n)';
  }
}
