import 'package:flutter/material.dart';
import 'package:flutter_cuda/appConfig.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';



void _showErrorDialog(BuildContext context, String message) {
  showDialog(
    context: context,
    builder: (BuildContext context) {
      return AlertDialog(
        title: Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
            },
            child: Text('OK'),
          ),
        ],
      );
    },
  );
}

void _showSuccessDialog(BuildContext context, String message) {
  showDialog(
    context: context,
    builder: (BuildContext context) {
      return AlertDialog(
        title: Text('Éxito'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
            },
            child: Text('OK'),
          ),
        ],
      );
    },
  );
}


class RegisterPage extends StatelessWidget {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  final TextEditingController passwordConfirmController = TextEditingController();
  final TextEditingController usernameController = TextEditingController();

  Future<void> _register(BuildContext context) async {
    String email = emailController.text;
    String password = passwordController.text;
    String confirmPassword = passwordConfirmController.text;
    String username = usernameController.text;

    if (password != confirmPassword) {
      _showErrorDialog(context, 'Las contraseñas no coinciden');
      return;
    }

    if (email.isNotEmpty && password.isNotEmpty && confirmPassword.isNotEmpty && username.isNotEmpty) {
      try {
        var response = await postRequest(email, password, username);
        Navigator.pushReplacementNamed(context, '/login');
        _showSuccessDialog(context, 'Registro exitoso');
        print('Respuesta del servidor: ${response.body}');
      } catch (e) {
        _showErrorDialog(context, 'Error durante la autenticación: $e');
      }
    } else {
      _showErrorDialog(context, 'Por favor, complete todos los campos');
    }
  }

  Future<http.Response> postRequest(String email, String password, String username) async {
    var url ='${AppConfig.apiUrl}:${AppConfig.port}/register/';

    Map<String, String> data = {
      'email': email,
      'password': password,
      'username':username
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
            'Registro de Usuario',
            style: TextStyle(
              fontSize: 24.0,
              color: Colors.yellow,
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
                  'Registro de usuario',
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    decoration: TextDecoration.underline,
                    color: Colors.black54
                  ),
                ),
                SizedBox(height: 20),
                TextField(
                  controller: usernameController,
                  decoration: InputDecoration(
                    labelText: 'Nombre de usuario',
                    hintText: 'Ingrese su username',
                    border: OutlineInputBorder(),
                  ),
                ),
                SizedBox(height: 20),
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
                  obscureText: true,
                  controller: passwordController,
                  decoration: InputDecoration(
                    labelText: 'Contraseña',
                    hintText: 'Ingrese su contraseña',
                    border: OutlineInputBorder(),
                  ),
                ),
                SizedBox(height: 16.0),
                TextField(
                  obscureText: true,
                  controller: passwordConfirmController,
                  decoration: InputDecoration(
                    labelText: 'Confirmar Contraseña',
                    hintText: 'Confirme su contraseña',
                    border: OutlineInputBorder(),
                  ),
                ),
                SizedBox(height: 16.0),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color:Colors.yellow),
                    minimumSize: Size(175, 50),  
                  ),
                  onPressed:() => _register(context),
                  child: Text(
                    'Registrar',
                    style: TextStyle(fontSize: 18.0,color:Colors.yellow),
                  ),
                ),
                SizedBox(height: 16.0),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color:Colors.yellow),
                    minimumSize: Size(175, 50),  
                  ),
                  onPressed: () {
                    Navigator.pop(context);
                    print('Cancelar');
                  },
                  child: Text(
                    'Cancelar',
                    style: TextStyle(fontSize: 18.0,color:Colors.yellow),
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
