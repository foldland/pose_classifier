import 'repetition_counter.dart';

/// An enumeration of supported exercise pose types for repetition counting.
///
/// Each pose type corresponds to a specific exercise movement that can be
/// detected and counted. The class names match the labels in the pose samples
/// file used for detection.
enum PoseType {
  /// A pushup exercise pose, detected at the "down" position
  pushups('pushups_down'),

  /// A squat exercise pose, detected at the "down" position
  squats('squats_down');

  /// Creates a pose type with the specified class name.
  const PoseType(this.className);

  /// The string identifier used to match this pose type in the pose samples
  /// file.
  final String className;

  /// Creates repetition counters for all available pose types.
  ///
  /// Returns a list of [RepetitionCounter] objects, one for each pose type,
  /// initialized with their corresponding class names.
  static List<RepetitionCounter> get repCounters =>
      values.map((pose) => RepetitionCounter(pose.className)).toList();
}
