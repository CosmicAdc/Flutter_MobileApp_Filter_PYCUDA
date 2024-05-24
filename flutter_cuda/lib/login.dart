import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'appConfig.dart'; // Importa la clase AppConfig

class LoginPage extends StatelessWidget {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();

  Future<void> _login(BuildContext context) async {
    String email = emailController.text;
    String password = passwordController.text;

    try {
      var response = await postRequest(context, email, password);
      if (response.statusCode == 200) {
        var responseData = json.decode(response.body);
        int id = responseData['id'];

        Navigator.pushNamed(
          context,
          '/home',
          arguments: {'id': id},
        );
      } else {
        _showErrorSnackbar(context, 'Error en la respuesta del servidor: ${response.statusCode}');
      }
    } catch (e) {
      _showErrorSnackbar(context, 'Error durante la autenticación: $e');
    }
  }

  void _showErrorSnackbar(BuildContext context, String message) {
    final snackBar = SnackBar(
      content: Text(message),
      backgroundColor: Colors.red,
    );
    ScaffoldMessenger.of(context).showSnackBar(snackBar);
  }

  Future<void> _showSettingsDialog(BuildContext context) async {
    String newUrl = AppConfig.apiUrl;
    int newPort = AppConfig.port;

    await showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Configuración de conexión'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
           children: [
              TextField(
                onChanged: (value) {
                  newUrl = 'http://'+value;
                },
                decoration: InputDecoration(labelText: 'Nueva IP'),
                keyboardType: TextInputType.numberWithOptions(decimal: true),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'[\d.]')),
                ],
              ),
              TextField(
                onChanged: (value) {
                  newPort = int.parse(value);
                },
                decoration: InputDecoration(labelText: 'Nuevo puerto'),
                keyboardType: TextInputType.number,
                inputFormatters: [
                  FilteringTextInputFormatter.digitsOnly,
                ],
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: Text('Cancelar'),
            ),
            ElevatedButton(
              onPressed: () {
                AppConfig.apiUrl = newUrl;
                AppConfig.port = newPort;
                Navigator.pop(context);
              },
              child: Text('Guardar'),
            ),
          ],
        );
      },
    );
  }

  Future<http.Response> postRequest(BuildContext context, String email, String password) async {
    var url = '${AppConfig.apiUrl}:${AppConfig.port}/login';

    Map<String, String> data = {
      'email': email,
      'password': password,
    };
    var body = json.encode(data);
    
    try {
      var response = await http.post(
        Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: body,
      );
      
      if (response.statusCode == 307) {
        var redirectedUrl = response.headers['location'];
        if (redirectedUrl != null) {
          response = await http.post(
            Uri.parse(redirectedUrl),
            headers: {"Content-Type": "application/json"},
            body: body,
          );
        }
      }
      
      return response;
    } catch (e) {
      _showErrorSnackbar(context, 'No se encontró la red: $e');
      rethrow; // Vuelve a lanzar la excepción para manejarla en _login
    }
  }

  @override
  Widget build(BuildContext context) {
    var url = '${AppConfig.apiUrl}:${AppConfig.port}';
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        title: Center(
          child: Text(
            url,
            style: TextStyle(
              fontSize: 24.0,
              color: Colors.yellow,
            ),
          ),
        ),
        backgroundColor: Colors.blue,
        actions: [
          IconButton(
            icon: Icon(Icons.settings),
            onPressed: () => _showSettingsDialog(context),
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue, Colors.black],
          ),
        ),
        child: Center(
          child: Container(
            width: MediaQuery.of(context).size.width,
            padding: EdgeInsets.all(20.0),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(10.0),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  spreadRadius: 10,
                  blurRadius: 10,
                  offset: Offset(0, 5),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: <Widget>[
                Image.network(
                  'https://upload.wikimedia.org/wikipedia/commons/b/b0/Logo_Universidad_Politécnica_Salesiana_del_Ecuador.png',
                  width: 300,
                  height: 200,
                ),
                SizedBox(height: 20),
                Text(
                  'Inicie sesión',
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    decoration: TextDecoration.underline,
                    color: Colors.black54
                  ),
                ),
                SizedBox(height: 16.0),
                TextField(
                  controller: emailController,
                  decoration: InputDecoration(
                    labelText: 'Correo',
                    hintText: 'Ingrese su correo',
                    border: OutlineInputBorder(),
                  ),
                ),
                SizedBox(height: 16.0),
                TextField(
                  controller: passwordController,
                  obscureText: true,
                  decoration: InputDecoration(
                    labelText: 'Contraseña',
                    hintText: 'Ingrese su contraseña',
                    border: OutlineInputBorder(),
                  ),
                ),
                SizedBox(height: 20.0),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                    ),
                    minimumSize: Size(175, 50), 
                  ),
                  onPressed: () => _login(context),
                  child: Text(
                    'Iniciar sesión',
                    style: TextStyle(fontSize: 18.0,color: Colors.yellow),
                    
                  ),
                ),
                SizedBox(height: 16.0),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                      
                    ),
                  minimumSize: Size(175, 50), 

                  ),
                  onPressed: () {
                    Navigator.pushNamed(context, '/register');
                    print('Nuevo? Registrate');
                  },
                  child: Text(
                    'Nuevo? Registrate',
                    style: TextStyle(fontSize: 18.0,color: Colors.yellow,),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
