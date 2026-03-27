import 'dart:math';

import 'package:collection/collection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;
import 'package:vector_math/vector_math_64.dart';

import 'classification_result.dart';
import 'pose_embedding.dart';
import 'pose_sample.dart';
import 'utils.dart';

/// A classifier for poses based on a collection of [PoseSample]s.
///
/// This implementation is inspired by the K-Nearest Neighbors algorithm
/// with outlier filtering. For more details, see:
/// https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
class PoseClassifier {
  /// Creates a [PoseClassifier] with the given [poseSamples].
  ///
  /// The [maxDistanceTopK] and [meanDistanceTopK] parameters define the
  /// number of top samples to consider for distance calculations. The
  /// [axesWeights] parameter allows customization of the weight for each
  /// axis, defaulting to [Vector3(1, 1, 0.2)].
  PoseClassifier(
    List<PoseSample> poseSamples, {
    int maxDistanceTopK = 30,
    int meanDistanceTopK = 10,
    Vector3? axesWeights,
  }) : _poseSamples = poseSamples,
       _maxDistanceTopK = maxDistanceTopK,
       _meanDistanceTopK = meanDistanceTopK,
       // Note Z has a lower weight as it is generally less accurate than X & Y.
       _axesWeights = axesWeights ?? Vector3(1, 1, 0.2);

  final List<PoseSample> _poseSamples;
  final int _maxDistanceTopK;
  final int _meanDistanceTopK;
  final Vector3 _axesWeights;

  /// Extracts the 3D landmarks from a given [pose].
  ///
  /// Returns a [LandmarkMap] containing the landmarks mapped by their type.
  static LandmarkMap extractPoseLandmarks(mlkit.Pose pose) {
    return pose.landmarks.map(
      (type, landmark) => MapEntry(type, landmark.position3D),
    );
  }

  /// Returns the maximum range of confidence values.
  ///
  /// This range is determined by the minimum of [_maxDistanceTopK] and
  /// [_meanDistanceTopK], which represent the number of samples used for
  /// confidence calculation.
  int confidenceRange() {
    return min(_maxDistanceTopK, _meanDistanceTopK);
  }

  /// Classifies the pose represented by the given [pose].
  ///
  /// Returns a [ClassificationResult] based on the landmarks extracted from
  /// the pose.
  ClassificationResult classifyFromPose(mlkit.Pose pose) {
    return classify(extractPoseLandmarks(pose));
  }

  /// Classifies the pose based on the provided [landmarks].
  ///
  /// Returns a [ClassificationResult] that reflects the confidence of each
  /// class based on the landmarks.
  ClassificationResult classify(LandmarkMap landmarks) {
    final result = ClassificationResult();
    // Return early if no landmarks detected.
    if (landmarks.isEmpty) {
      return result;
    }

    final xAxis = Vector3(-1, 1, 1);
    // We do flipping on X-axis so we are horizontal (mirror) invariant.
    final flippedLandmarks = landmarks.map(
      (type, landmark) => MapEntry(type, landmark.multiplyVector(xAxis)),
    );

    final embedding = PoseEmbedding.getPoseEmbedding(landmarks);
    final flippedEmbedding = PoseEmbedding.getPoseEmbedding(flippedLandmarks);

    // Classification is done in two stages:
    //  * First we pick top-K samples by MAX distance. It allows to remove
    //    samples that are almost the same as given pose, but maybe has few
    //    joints bent in the other direction.
    //  * Then we pick top-K samples by MEAN distance. After outliers are
    //    removed, we pick samples that are closest by average.

    // Keeps max distance on top so we can pop it when top_k size is reached.
    final maxDistances = PriorityQueue<(PoseSample, double)>(
      (o1, o2) => -o1.$2.compareTo(o2.$2),
    );
    // Retrieve top K poseSamples by least distance to remove outliers.
    for (final poseSample in _poseSamples) {
      final sampleEmbedding = poseSample.embedding;

      var originalMax = 0.0;
      var flippedMax = 0.0;
      for (var i = 0; i < embedding.length; i++) {
        originalMax = max(
          originalMax,
          // TODO(nikolas.rimikis): perf; remove copy
          (embedding[i] - sampleEmbedding[i])
              .multiplyVector(_axesWeights)
              .maxAbs(),
        );
        flippedMax = max(
          flippedMax,
          (flippedEmbedding[i] - sampleEmbedding[i])
              .multiplyVector(_axesWeights)
              .maxAbs(),
        );
      }
      // Set the max distance as min of original and flipped max distance.
      maxDistances.add((poseSample, min(originalMax, flippedMax)));
      // We only want to retain top n so pop the highest distance.
      if (maxDistances.length > _maxDistanceTopK) {
        maxDistances.removeFirst();
      }
    }

    // Keeps higher mean distances on top so we can pop it when top_k size is
    // reached.
    final meanDistances = PriorityQueue<(PoseSample, double)>(
      (o1, o2) => -o1.$2.compareTo(o2.$2),
    );
    // Retrieve top K poseSamples by least mean distance to remove outliers.
    for (final sampleDistances in maxDistances.toList()) {
      final poseSample = sampleDistances.$1;
      final sampleEmbedding = poseSample.embedding;

      var originalSum = 0.0;
      var flippedSum = 0.0;
      for (var i = 0; i < embedding.length; i++) {
        originalSum += (embedding[i] - sampleEmbedding[i])
            .multiplyVector(_axesWeights)
            .sumAbs();
        flippedSum += (flippedEmbedding[i] - sampleEmbedding[i])
            .multiplyVector(_axesWeights)
            .sumAbs();
      }
      // Set the mean distance as min of original and flipped mean distances.
      final meanDistance =
          min(originalSum, flippedSum) / (embedding.length * 2);
      meanDistances.add((poseSample, meanDistance));
      // We only want to retain top k so pop the highest mean distance.
      if (meanDistances.length > _meanDistanceTopK) {
        meanDistances.removeFirst();
      }
    }

    for (final sampleDistances in meanDistances.toList()) {
      final className = sampleDistances.$1.className;
      result.incrementClassConfidence(className);
    }

    return result;
  }
}
