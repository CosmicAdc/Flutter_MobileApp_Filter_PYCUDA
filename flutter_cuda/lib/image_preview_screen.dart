import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class ImagePreviewScreen extends StatefulWidget {
  final XFile imageFile;
  final String pathOriginal;

  ImagePreviewScreen({required this.imageFile, required this.pathOriginal});

  @override
  _ImagePreviewScreenState createState() => _ImagePreviewScreenState();
}

class _ImagePreviewScreenState extends State<ImagePreviewScreen> {
  String? _filteredImagePath;
  Widget _buildFilterButton(String filterName, String assetPath, VoidCallback? onPressed) {
    return GestureDetector(
      onTap: onPressed,
      child: Column(
        children: [
          CircleAvatar(
            backgroundImage: AssetImage(assetPath),
            radius: 20,
          ),
          SizedBox(height: 4),
          Text(
            filterName,
            style: TextStyle(
              color: Colors.white
            ),
          ),
        ],
      ),
    );
  }

Future<void> aplicarFiltro(String servicioFiltro) async {
  final urlBase = 'http://10.0.2.2:8000/';
  final url = Uri.parse('$urlBase$servicioFiltro/');
  final headers = {'Content-Type': 'application/json'};
  final body = jsonEncode({
    'path_file': widget.pathOriginal,
  });

  try {
    final response = await http.post(url, headers: headers, body: body);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final preRuta = data['ruta_imagen'];
      final rutaImagenFiltrada = 'http://10.0.2.2:8000/${preRuta.substring(4)}';

      setState(() {
        _filteredImagePath = rutaImagenFiltrada;
      });

      print('Ruta de la imagen filtrada: $rutaImagenFiltrada');
    } else {
      print('Error al aplicar el filtro Circulo: ${response.statusCode}');
    }
  } catch (e) {
    print('Error al aplicar el filtro Circulo: $e');
  }
}


Future<void> aplicarOriginal() async {
      setState(() {
        _filteredImagePath = null;
      });
}



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Vista previa'),
      ),
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: _filteredImagePath != null
                    ? NetworkImage(_filteredImagePath!)
                    : FileImage(File(widget.imageFile.path)),
                fit: BoxFit.cover,
              ),
            ),
          ),
          Positioned(
            bottom: 20,
            left: 20,
            child: IconButton(
              icon: Icon(Icons.arrow_back),
              onPressed: () {
                Navigator.pop(context);
              },
            ),
          ),
          Positioned(
            bottom: 100,
            left: 20,
            child: Row(
              children: [
               _buildFilterButton(
                  'Original',
                  'assets/original.jpg',
                  () async {
                    await aplicarOriginal();
                  },
                ),
                SizedBox(width: 16),
                _buildFilterButton(
                  'UPS',
                  'assets/ups.jpg',
                  () async {
                    await aplicarFiltro('filtroUPS');
                  },
                ),
                SizedBox(width: 16),
                _buildFilterButton(
                  'Marea Roja',
                  'assets/marea.jpg',
                  () async {
                    await aplicarFiltro('filtroMarea');
                  },
                ),
                SizedBox(width: 16),
                _buildFilterButton(
                  'Circulo',
                  'assets/circulo.jpg',
                  () async {
                    await aplicarFiltro('filtroCirculo');
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}