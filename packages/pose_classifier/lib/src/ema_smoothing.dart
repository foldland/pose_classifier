import 'dart:collection';

import 'package:meta/meta.dart';
import 'classification_result.dart';
import 'pose_classifier.dart' show PoseClassifier;

/// A class that implements Exponential Moving Average (EMA) smoothing
/// over a window of pose classification results.
///
/// This class maintains a sliding window of classification results and
/// computes a smoothed result based on the specified window size and
/// smoothing factor (alpha).
class EMASmoothing {
  /// Creates an instance of [EMASmoothing].
  ///
  /// The [windowSize] parameter defines the size of the sliding window,
  /// and [alpha] defines the smoothing factor. The default values are
  /// 10 for [windowSize] and 0.2 for [alpha].
  EMASmoothing({int windowSize = 10, double alpha = 0.2})
    : _alpha = alpha,
      _windowSize = windowSize;

  static const Duration _resetThreshold = Duration(milliseconds: 100);

  final int _windowSize;
  final double _alpha;

  /// A window of [ClassificationResult]s as outputted by the
  /// [PoseClassifier].
  ///
  /// Smoothing is performed over this window of size [_windowSize].
  final DoubleLinkedQueue<ClassificationResult> _window = DoubleLinkedQueue();

  /// The timestamp of the last input received.
  ///
  /// This attribute is used to determine if the input is too far away
  /// from the previous one in time, which may trigger a reset of the
  /// sliding window.
  @visibleForTesting
  DateTime? lastInput;

  /// Returns a smoothed [ClassificationResult] based on the provided
  /// [classificationResult].
  ///
  /// This method resets the memory if the input is too far away from the
  /// previous one in time, and it updates the sliding window with the new
  /// classification result.
  ClassificationResult getSmoothedResult(
    ClassificationResult classificationResult,
  ) {
    // Resets memory if the input is too far away from the previous one in time.
    final lastInput = this.lastInput;
    final now = DateTime.now();
    if (lastInput != null && now.difference(lastInput) > _resetThreshold) {
      _window.clear();
    }
    this.lastInput = now;

    // If we are at window size, remove the last (oldest) result.
    if (_window.length == _windowSize) {
      _window.removeLast();
    }
    // Insert at the beginning of the window.
    _window.addFirst(classificationResult);

    final allClasses = <String>{};
    for (final result in _window) {
      allClasses.addAll(result.getAllClasses());
    }

    final smoothed = ClassificationResult();
    for (final className in allClasses) {
      var factor = 1.0;
      var topSum = 0.0;
      var bottomSum = 0.0;
      for (final result in _window) {
        final value = result.getClassConfidence(className);

        topSum += factor * value;
        bottomSum += factor;

        factor = factor * (1.0 - _alpha);
      }
      smoothed.setClassConfidence(className, topSum / bottomSum);
    }
    return smoothed;
  }
}
