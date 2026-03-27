/// Result of a pose classification.
class PoseClassificationResult {
  /// Creates a new pose classification result.
  const PoseClassificationResult({
    required this.className,
    required this.confidence,
  });

  /// The class of classified pose.
  final String? className;

  /// The confidence of the classified [className].
  final double? confidence;

  @override
  String toString() {
    return '''
PoseClassificationResult(
  className: $className,
  confidence: $confidence,
)
''';
  }
}

/// Result of a streaming pose classification.

class StreamingClassificationResult extends PoseClassificationResult {
  /// Creates a new pose classification result for a streaming processor.
  const StreamingClassificationResult({
    required this.repetitionClass,
    required this.repetitionCount,

    super.className,
    super.confidence,
  });

  /// The detected pose repetition.
  final String? repetitionClass;

  /// The number of repetitions of the detected [repetitionClass].
  final int? repetitionCount;

  @override
  String toString() {
    return '''
StreamingClassificationResult(
  repetitionClass: $repetitionClass,
  repetitionCount: $repetitionCount,
  lastPose: ($className, $confidence),
)
''';
  }
}
