typedef Repetition = ({String name, int reps});
typedef PoseConfidence = ({String name, double confidence});

class PoseClassificationResult {
  PoseClassificationResult({
    required Repetition? repetitions,
    required this.lastPose,
  }) : repetitionClass = repetitions?.name,
       repetitionCount = repetitions?.reps;

  /// The detected pose reppetition.
  final String? repetitionClass;

  /// The number of repititions.
  final int? repetitionCount;

  /// The last detected pose with its confidence.
  final PoseConfidence? lastPose;

  @override
  String toString() {
    return '''
PoseClassificationResult(
  repetitionClass: $repetitionClass,
  repetitionCount: $repetitionCount,
  lastPose: $lastPose,
)
''';
  }
}
