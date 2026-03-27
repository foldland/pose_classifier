import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:pose_classifier/pose_classifier.dart';

void main() {
  runApp(const MyApp());
}

/// The main App.
class MyApp extends StatelessWidget {
  /// Constructs a [MyApp].
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Detector Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.greenAccent),
      ),
      home: const HomeScreen(),
    );
  }
}

/// The home screen.
class HomeScreen extends StatefulWidget {
  /// Constructs a [HomeScreen].
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreen();
}

class _HomeScreen extends State<HomeScreen> {
  late PoseDetector poseDetector;
  late CameraController controller;
  late List<CameraDescription> cameras;

  late PoseClassifierProcessor poseClassifierProcessor;
  bool initialized = false;

  // drop frames, when busy
  bool lock = false;

  PoseClassificationResult? result;
  InputImageRotation? rotation;

  @override
  void initState() {
    super.initState();

    poseDetector = PoseDetector(options: PoseDetectorOptions());
    unawaited(initPoseDetection().then((_) => initCamera()));
  }

  @override
  void dispose() {
    controller.removeListener(updatePreset);

    unawaited(
      controller.stopImageStream().then(
        (_) => controller.dispose().then((_) => poseDetector.close()),
      ),
    );
    super.dispose();
  }

  Future<void> initPoseDetection() async {
    try {
      final csv = await rootBundle.loadString(
        'assets/poses/fitness_pose_samples.csv',
      );
      final csvLines = const LineSplitter().convert(csv);
      final poseSamples = loadSamples(csvLines);
      poseClassifierProcessor = PoseClassifierProcessor(
        poseSamples: poseSamples,
      );
    } on Exception catch (e) {
      debugPrint('Error when loading pose samples.\n$e');
    }
  }

  Future<void> initCamera() async {
    try {
      cameras = await availableCameras();

      controller = CameraController(
        cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );

      controller.addListener(updatePreset);

      await controller.initialize();
      initialized = true;
      setState(() {});

      unawaited(handleStream());
    } on CameraException catch (e) {
      switch (e.code) {
        case 'CameraAccessDenied':
          // Handle access errors here.
          break;
        default:
          // Handle other errors here.
          break;
      }
    }
  }

  Future<void> switchCamera() async {
    if (!initialized) {
      return;
    }

    final index = (controller.cameraId + 1) % cameras.length;
    await controller.setDescription(cameras[index]);
  }

  void updatePreset() {
    final camera = controller.description;
    var sensorOrientation = camera.sensorOrientation;
    if (Platform.isAndroid) {
      final compensation = switch (controller.value.deviceOrientation) {
        DeviceOrientation.portraitUp => 0,
        DeviceOrientation.landscapeLeft => 90,
        DeviceOrientation.portraitDown => 180,
        DeviceOrientation.landscapeRight => 270,
      };

      if (camera.lensDirection == CameraLensDirection.front) {
        sensorOrientation = (sensorOrientation + compensation) % 360;
      } else {
        sensorOrientation = (sensorOrientation - compensation + 360) % 360;
      }
    }

    rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
  }

  InputImage? convertImage(CameraImage image) {
    final rotation = this.rotation;
    if (rotation == null) {
      return null;
    }

    final format = Platform.isAndroid
        ? InputImageFormat.nv21
        : InputImageFormat.bgra8888;

    assert(
      InputImageFormatValue.fromRawValue(image.format.raw as int) == format,
      'The image format must be either NV21 or BGRA8888',
    );

    // NV21 and BGRA8888 always have a single plane
    final plane = image.planes.single;

    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  Future<void> detectPose(CameraImage cameraImage) async {
    if (lock) {
      return;
    }
    lock = true;

    final inputImage = convertImage(cameraImage);
    if (inputImage == null) {
      return;
    }

    // classify pose
    final poses = await poseDetector.processImage(inputImage);
    if (!mounted) {
      return;
    }

    if (poses.isNotEmpty) {
      result = poseClassifierProcessor.classifyPose(poses.first);
    }

    lock = false;
    setState(() {});
  }

  Future<void> handleStream() async {
    if (!controller.supportsImageStreaming()) {
      throw UnimplementedError('Set an error message when unsupported');
    }

    await controller.startImageStream(detectPose);
  }

  @override
  Widget build(BuildContext context) {
    Widget? cameraPreview;
    if (initialized) {
      cameraPreview = CameraPreview(
        controller,
        child: result != null
            ? Center(
                child: Container(
                  padding: const EdgeInsets.all(8),
                  color: Colors.white,
                  child: Text(result.toString()),
                ),
              )
            : null,
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Flutter Pose Detector')),
      body: cameraPreview,
      floatingActionButton: FloatingActionButton(
        onPressed: switchCamera,
        child: const Icon(Icons.cameraswitch),
      ),
    );
  }
}
