import 'dart:convert';

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;
import 'package:vector_math/vector_math_64.dart';

import 'pose_embedding.dart';
import 'utils.dart';

/// Parses a list of CSV lines as pose samples and returns their values.
///
/// Throws a [FormatException] if the [csvLines] strings do not
/// specify a valid [PoseSample].
///
/// Use a [LineSplitter] to convert a CSV string into the lines.
///
/// The expected format is:
/// ```csv
/// Name,Class,X1,Y1,Z1,X2,Y2,Z2...
/// Name,Class,X1,Y1,Z1,X2,Y2,Z2...
/// Name,Class,X1,Y1,Z1,X2,Y2,Z2...
/// ```
///
/// Example:
/// ```dart
/// final csv = await rootBundle.loadString('assets/fitness_pose_samples.csv');
/// final csvLines = const LineSplitter().convert(csv);
/// final poseSamples = loadSamples(csvLines);
/// ```
List<PoseSample> loadSamples(List<String> csvLines) {
  final poseSamples = <PoseSample>[];

  for (final line in csvLines) {
    final sample = PoseSample.parse(line);

    poseSamples.add(sample);
  }

  return poseSamples;
}

/// Represents a pose sample for classification.
///
/// Classifies [mlkit.Pose] based on given [PoseSample]s using an
/// algorithm inspired by K-Nearest Neighbors with outlier filtering.
/// See: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
class PoseSample {
  /// Creates a [PoseSample] instance from the given parameters.
  PoseSample._(this.name, this.className, LandmarkMap landmarks)
    : embedding = PoseEmbedding.getPoseEmbedding(landmarks);

  /// Parses a CSV line as a pose sample and returns its value.
  ///
  /// Throws a [FormatException] if the [csvLine] string does not have
  /// the exact number of tokens needed.
  ///
  /// The expected format is: Name,Class,X1,Y1,Z1,X2,Y2,Z2...
  factory PoseSample.parse(String csvLine, [String separator = ',']) {
    final tokens = csvLine.split(separator);

    // Format is expected to be Name,Class,X1,Y1,Z1,X2,Y2,Z2...
    // + 2 is for Name & Class.
    if (tokens.length !=
        (mlkit.PoseLandmarkType.values.length * _numDims) + 2) {
      throw const FormatException('Invalid number of tokens for landmark');
    }

    final name = tokens[0];
    final className = tokens[1];
    final landmarks = <mlkit.PoseLandmarkType, Vector3>{};

    for (final landmark in mlkit.PoseLandmarkType.values) {
      // Read from the third token; first 2 tokens are name and class.
      final i = (landmark.index * _numDims) + 2;

      try {
        final x = double.parse(tokens[i]);
        final y = double.parse(tokens[i + 1]);
        final z = double.parse(tokens[i + 2]);
        landmarks[landmark] = Vector3(x, y, z);
      } on FormatException catch (_) {
        throw FormatException(
          'Invalid value ${tokens[i]} for landmark position.',
        );
      }
    }

    return PoseSample._(name, className, landmarks);
  }

  /// Number of dimensions per landmark.
  static const int _numDims = 3;

  /// The name of the pose sample.
  final String name;

  /// The class name of the pose sample.
  final String className;

  /// The embedding representation of the pose sample.
  final List<Vector3> embedding;
}
