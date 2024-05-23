import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class RegisterPage extends StatelessWidget {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  final TextEditingController passwordConfirmController = TextEditingController();
  final TextEditingController usernameController = TextEditingController();

  Future<void> _register(BuildContext context) async {
    String email = emailController.text;
    String password = passwordController.text;
    String confirmpassword = passwordConfirmController.text;
    String username = usernameController.text;

    if (password != confirmpassword) {
      // Mostrar un mensaje de error o realizar alguna acción
      print('Las contraseñas no coinciden');
      return;
    }
    try {
      var response = await postRequest(email, password, username);
      // ignore: use_build_context_synchronously
      Navigator.pushReplacementNamed(context, '/login');
      print('Respuesta del servidor: ${response.body}');

    } catch (e) {
      print('Error durante la autenticación: $e');
    }
  }

  Future<http.Response> postRequest(String email, String password, String username) async {
    var url ='http://10.0.2.2:8000/register/';

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
                  'Registro de usuario',
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    decoration: TextDecoration.underline,
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
                    textStyle: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),

                  ),
                  onPressed:() => _register(context),
                  child: Text(
                    'Registrar',
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
                    Navigator.pop(context);
                    print('Cancelar');
                  },
                  child: Text(
                    'Cancelar',
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
