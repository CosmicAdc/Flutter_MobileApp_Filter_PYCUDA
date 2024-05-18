import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  CameraScreen({required this.cameras});
  
  @override
  _CameraScreenState createState() => _CameraScreenState();
}



class _CameraScreenState extends State<CameraScreen> {

  Future<String?> uploadImage(XFile imageFile) async {
  final url = Uri.parse('http://10.0.2.2:8000/upload/');
  final request = http.MultipartRequest('POST', url);

  final file = await http.MultipartFile.fromPath(
    'files', // El nombre del campo de archivo en tu servicio FastAPI
    imageFile.path,
    filename: path.basename(imageFile.path),
  );

  request.files.add(file);

  try {
    final response = await request.send();
    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      final responseData = json.decode(responseBody);
      final nameFile = responseData['name_file'];
      print('Imagen subida exitosamente. Nombre de la ruta: $nameFile');
      return nameFile;
    } else {
      return null;
      print('Error al subir la imagen: ${response.reasonPhrase}');
    }
  } catch (e) {
    print('Error al subir la imagen: $e');
  }

}



  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  XFile? _capturedImage;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

   Future<void> _initializeCamera() async {
    final firstCamera = widget.cameras.first; // Usa widget.cameras aquí
    _controller = CameraController(
      firstCamera,
      ResolutionPreset.medium,
    );

    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

@override
Widget build(BuildContext context) {
  return Scaffold(
    appBar: AppBar(title: Text('Cámara')),
    body: Column(
      children: [
        Expanded(
          child: FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                if (_controller.value.isInitialized) {
                  return AspectRatio(
                    aspectRatio: _controller.value.aspectRatio,
                    child: CameraPreview(_controller),
                  );
                } else {
                  return Center(child: CircularProgressIndicator());
                }
              } else {
                return Center(child: CircularProgressIndicator());
              }
            },
          ),
        ),
                if (_capturedImage != null)
                  SizedBox(
                    height: 200,
                    width: 200, 
                    child: Image.file(
                      File(_capturedImage!.path),
                      fit: BoxFit.cover,
                       width: double.infinity, 
                    ),
            ),
              ],
            ),
        bottomNavigationBar: BottomAppBar(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            IconButton(
              icon: Icon(Icons.home),
              onPressed: () {
                 Navigator.pop(context);
              },
            ),
            IconButton(
              icon: Icon(Icons.play_circle),
               onPressed: () {
              Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => CameraScreen(cameras: widget.cameras),
                  ),
                );
              },
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        child: Icon(Icons.camera),
        onPressed: () async {
          try {
            final image = await _controller.takePicture();
            _capturedImage = image;
            setState(() {});
            final response= await uploadImage(image);

            print('Imagen capturada: ${image.path}');
          } catch (e) {
            print('Error al tomar la foto: $e');
          }
        },
      ),
    );
  }
}