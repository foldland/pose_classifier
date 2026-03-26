// dart format width=120

import 'dart:math' as math;

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart' as mlkit;
import 'package:meta/meta.dart';
import 'package:vector_math/vector_math_64.dart';

import 'utils.dart';

/// A class that generates embeddings for a given list of pose landmarks.
class PoseEmbedding {
  const PoseEmbedding._();

  /// Multiplier to apply to the torso to get minimal body size.
  /// Picked this by experimentation.
  static const double _torsoMultiplier = 2.5;

  /// Generates a pose embedding from the provided [landmarks].
  ///
  /// Normalizes the landmarks and computes the embedding based on the
  /// normalized values.
  static List<Vector3> getPoseEmbedding(LandmarkMap landmarks) {
    final normalized = normalize(landmarks);
    return getEmbedding(normalized);
  }

  /// Normalizes the given [landmarks] by translating and scaling them.
  ///
  /// Translation normalization centers the landmarks around the midpoint
  /// of the hips, and scaling is done based on the size of the pose.
  @visibleForTesting
  static LandmarkMap normalize(LandmarkMap landmarks) {
    // Normalize translation.
    final center = Vector3Utils.average(
      landmarks[mlkit.PoseLandmarkType.leftHip]!,
      landmarks[mlkit.PoseLandmarkType.rightHip]!,
    );

    final normalized = <mlkit.PoseLandmarkType, Vector3>{};
    for (final MapEntry(:key, :value) in landmarks.entries) {
      normalized[key] = value - center;
    }

    // Normalize scale.
    final poseSize = getPoseSize(normalized);
    var scale = 1.0 / poseSize;
    // Multiplication by 100 is not required, but makes it easier to debug.
    scale *= 100;
    for (final MapEntry(:key, :value) in normalized.entries) {
      normalized[key] = value * scale;
    }

    return normalized;
  }

  /// Computes the size of the pose based on the provided [landmarks].
  ///
  /// This method uses only 2D landmarks to compute the pose size, as using
  /// Z coordinates was not helpful in experimentation. The size is based on
  /// the distance between the hips and shoulders, scaled by the torso
  /// multiplier.
  @visibleForTesting
  static double getPoseSize(LandmarkMap landmarks) {
    // Note: This approach uses only 2D landmarks to compute pose size as using
    // Z wasn't helpful in our experimentation but you're welcome to tweak.
    final hipsCenter = Vector3Utils.average(
      landmarks[mlkit.PoseLandmarkType.leftHip]!,
      landmarks[mlkit.PoseLandmarkType.rightHip]!,
    );
    final shouldersCenter = Vector3Utils.average(
      landmarks[mlkit.PoseLandmarkType.leftShoulder]!,
      landmarks[mlkit.PoseLandmarkType.rightShoulder]!,
    );

    final torsoSize = (hipsCenter - shouldersCenter).l2Norm2D();

    // torsoSize * TORSO_MULTIPLIER is the floor we want based on
    // experimentation but actual size can be bigger for a given pose depending
    // on extension of limbs etc so we calculate that.
    final maxDistance = landmarks.values
        .map((landmark) => (hipsCenter - landmark).l2Norm2D())
        .fold(torsoSize * _torsoMultiplier, math.max);

    return maxDistance;
  }

  /// Generates a pose embedding from the given normalized landmarks [lm].
  ///
  /// This method computes several pairwise 3D distances to form the pose
  /// embedding. The distances are selected based on experimentation for
  /// best results with default pose classes.
  @visibleForTesting
  static List<Vector3> getEmbedding(LandmarkMap lm) {
    // We group our distances by number of joints between the pairs.
    // One joint.
    final embedding = <Vector3>[
      (Vector3Utils.average(lm[mlkit.PoseLandmarkType.leftHip]!, lm[mlkit.PoseLandmarkType.rightHip]!) -
          Vector3Utils.average(lm[mlkit.PoseLandmarkType.leftShoulder]!, lm[mlkit.PoseLandmarkType.rightShoulder]!)),

      (lm[mlkit.PoseLandmarkType.leftShoulder]! - lm[mlkit.PoseLandmarkType.leftElbow]!),
      (lm[mlkit.PoseLandmarkType.rightShoulder]! - lm[mlkit.PoseLandmarkType.rightElbow]!),

      (lm[mlkit.PoseLandmarkType.leftElbow]! - lm[mlkit.PoseLandmarkType.leftWrist]!),
      (lm[mlkit.PoseLandmarkType.rightElbow]! - lm[mlkit.PoseLandmarkType.rightWrist]!),

      (lm[mlkit.PoseLandmarkType.leftHip]! - lm[mlkit.PoseLandmarkType.leftKnee]!),
      (lm[mlkit.PoseLandmarkType.rightHip]! - lm[mlkit.PoseLandmarkType.rightKnee]!),

      (lm[mlkit.PoseLandmarkType.leftKnee]! - lm[mlkit.PoseLandmarkType.leftAnkle]!),
      (lm[mlkit.PoseLandmarkType.rightKnee]! - lm[mlkit.PoseLandmarkType.rightAnkle]!),

      // Two joints.
      (lm[mlkit.PoseLandmarkType.leftShoulder]! - lm[mlkit.PoseLandmarkType.leftWrist]!),
      (lm[mlkit.PoseLandmarkType.rightShoulder]! - lm[mlkit.PoseLandmarkType.rightWrist]!),

      (lm[mlkit.PoseLandmarkType.leftHip]! - lm[mlkit.PoseLandmarkType.leftAnkle]!),
      (lm[mlkit.PoseLandmarkType.rightHip]! - lm[mlkit.PoseLandmarkType.rightAnkle]!),

      // Four joints.
      (lm[mlkit.PoseLandmarkType.leftHip]! - lm[mlkit.PoseLandmarkType.leftWrist]!),
      (lm[mlkit.PoseLandmarkType.rightHip]! - lm[mlkit.PoseLandmarkType.rightWrist]!),

      // Five joints.
      (lm[mlkit.PoseLandmarkType.leftShoulder]! - lm[mlkit.PoseLandmarkType.leftAnkle]!),
      (lm[mlkit.PoseLandmarkType.rightShoulder]! - lm[mlkit.PoseLandmarkType.rightAnkle]!),

      (lm[mlkit.PoseLandmarkType.leftHip]! - lm[mlkit.PoseLandmarkType.leftWrist]!),
      (lm[mlkit.PoseLandmarkType.rightHip]! - lm[mlkit.PoseLandmarkType.rightWrist]!),

      // Cross body.
      (lm[mlkit.PoseLandmarkType.leftElbow]! - lm[mlkit.PoseLandmarkType.rightElbow]!),
      (lm[mlkit.PoseLandmarkType.leftKnee]! - lm[mlkit.PoseLandmarkType.rightKnee]!),

      (lm[mlkit.PoseLandmarkType.leftWrist]! - lm[mlkit.PoseLandmarkType.rightWrist]!),
      (lm[mlkit.PoseLandmarkType.leftAnkle]! - lm[mlkit.PoseLandmarkType.rightAnkle]!),
    ];

    return embedding;
  }
}
