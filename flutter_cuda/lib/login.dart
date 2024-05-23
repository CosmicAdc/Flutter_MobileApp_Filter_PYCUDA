import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class LoginPage extends StatelessWidget {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();

  Future<void> _login(BuildContext context) async {
    String email = emailController.text;
    String password = passwordController.text;

    try {
      var response = await postRequest(email, password);
      if (response.statusCode == 200) {
        // Desempaquetar la respuesta JSON
        var responseData = json.decode(response.body);

        // Acceder a los datos de la respuesta
        String message = responseData['message'];
        String user = responseData['user'];
        int id = responseData['id'];

        // Procesar los datos aquí
        print('Mensaje: $message');
        print('Usuario: $user');
        print('ID: $id');

        Navigator.pushNamed(
          context,
          '/home',
          arguments: {'id': id},
        );

      } else {
        // Manejar errores del servidor
        print('Error en la respuesta del servidor: ${response.statusCode}');
      }
    } catch (e) {
      print('Error durante la autenticación: $e');
    }
  }

  Future<http.Response> postRequest(String email, String password) async {
    var url ='http://192.168.0.102:8000/login/';

    Map<String, String> data = {
      'email': email,
      'password': password,
    };
    // Encode Map to JSON
    var body = json.encode(data);

    var response = await http.post(Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: body
    );
    return response;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        title: Center(
          child: Text(
            'Inicio de Sesión del Usuario',
            style: TextStyle(
              fontSize: 24.0,
              color: Colors.white,
            ),
          ),
        ),
        backgroundColor: Colors.blue,
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
            width: MediaQuery.of(context).size.width ,
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
                    height: 200
                ),
                SizedBox(height: 20),
                Text(
                  'Inicie sesión',
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    decoration: TextDecoration.underline,
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
                SizedBox(height: 16.0),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),

                  ),
                  onPressed:() => _login(context),
                  child: Text(
                    'Iniciar sesión',
                    style: TextStyle(fontSize: 18.0),
                  ),
                ),
                SizedBox(height: 16.0),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),

                  ),
                  onPressed: () {
                    Navigator.pushNamed(context, '/register');
                    print('Nuevo? Registrate');
                  },
                  child: Text(
                    'Nuevo? Registrate',
                    style: TextStyle(fontSize: 18.0),
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
