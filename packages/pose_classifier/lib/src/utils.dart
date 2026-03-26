import 'dart:math' as math;

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;
import 'package:vector_math/vector_math_64.dart';

/// A type alias for mapping pose landmarks to their 3D vector positions.
typedef LandmarkMap = Map<mlkit.PoseLandmarkType, Vector3>;

/// Provides utility methods for [Vector3] operations.
extension Vector3Utils on Vector3 {
  /// Multiplies this vector with [other] component-wise.
  ///
  /// Returns a new [Vector3] containing the result of the multiplication.
  Vector3 multiplyVector(Vector3 other) {
    return clone()..multiply(other);
  }

  /// Calculates the average of two vectors.
  ///
  /// Returns a new [Vector3] representing the midpoint between [a] and [b].
  static Vector3 average(Vector3 a, Vector3 b) {
    return a.clone()
      ..add(b)
      ..scale(0.5);
  }

  /// Calculates the 2D L2 norm (Euclidean distance) of this vector.
  ///
  /// Only considers the x and y components, ignoring z.
  /// Returns the square root of (x² + y²).
  double l2Norm2D() {
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2));
  }

  /// Returns the maximum absolute value among all components.
  ///
  /// Compares the absolute values of x, y, and z components and returns
  /// the largest one.
  double maxAbs() {
    return math.max(math.max(x.abs(), y.abs()), z.abs());
  }

  /// Calculates the sum of absolute values of all components.
  ///
  /// Returns |x| + |y| + |z|.
  double sumAbs() {
    return x.abs() + y.abs() + z.abs();
  }
}

/// Extension on [mlkit.PoseLandmark] to provide 3D position functionality.
extension PoseLandmarkVector on mlkit.PoseLandmark {
  /// Gets the 3D position of the landmark as a [Vector3].
  ///
  /// Creates a new [Vector3] from the landmark's x, y, and z coordinates.
  Vector3 get position3D {
    return Vector3(x, y, z);
  }
}
