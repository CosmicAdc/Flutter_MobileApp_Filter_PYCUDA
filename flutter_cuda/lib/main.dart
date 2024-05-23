import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_cuda/Posts.dart';
import 'package:flutter_cuda/appConfig.dart';
import 'camera_screen.dart';
import 'login.dart';
import 'register.dart';
import 'package:http/http.dart' as http;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();

  runApp(MyApp(cameras: cameras));
}

Future<List<Post>> fetchPosts() async {
  final response = await http.get(Uri.parse('${AppConfig.apiUrl}:${AppConfig.port}/posts/'));
  
  if (response.statusCode == 200) {
    // Si la solicitud es exitosa, analiza el JSON
    List<dynamic> data = json.decode(response.body);
    List<Post> posts = data.map((json) => Post.fromJson(json)).toList();
for (var post in posts) {
      print('Post ID: ${post.id}');
      print('User ID: ${post.userId}');
      print('Image Path: ${post.imagePath}');
      print('Description: ${post.description}');
      print('username: ${post.username}');
      print('----------------------');
    }
    return posts;
  } else {
    // Si la solicitud falla, lanza una excepción
    throw Exception('Failed to load posts');
  }
}





class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home:  LoginPage(),
      routes: {
        '/register': (context) => RegisterPage(), // Define la ruta para RegisterPage
        '/home': (context) => MyHomePage(cameras: cameras),
        '/login': (context) => LoginPage(),
      },
    );
  }
}

class MyHomePage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const MyHomePage({Key? key, required this.cameras}) : super(key: key);

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int? userId;
  List<Post> _posts = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _fetchPosts();
  }

  Future<void> _fetchPosts() async {
    setState(() {
      _isLoading = true;
    });
    try {
      final posts = await fetchPosts();
      setState(() {
        _posts = posts;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      // Handle error
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final arguments = ModalRoute.of(context)?.settings.arguments as Map?;
    userId = arguments?['id'] as int?;
    print(userId);
  }

  @override
  Widget build(BuildContext context) {
    final Map<String, dynamic> arguments =
        ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>;
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Insta-UPS',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.yellow
          ),
        ),
        backgroundColor: Colors.blueAccent,
        centerTitle: true,
      ),
      body: RefreshIndicator(
        onRefresh: _fetchPosts,
        child: ListView.separated(
          itemCount: _posts.length + 1, // Agregamos 1 para el botón de carga
          separatorBuilder: (context, index) => Divider(),
            itemBuilder: (context, index) {
              if (index == _posts.length) {
                return _isLoading
                    ? _buildLoadingIndicator() // Muestra el indicador de carga si está cargando
                    : _buildLoadMoreButton(); // Muestra el botón de carga si no está cargando
              } else {
                return Container(
                  padding: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(1),
                        spreadRadius: 2,
                        blurRadius: 7,
                        offset: Offset(0, 3),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Align(
                        alignment: Alignment.center,
                        child: Text(
                          _posts[index].username,
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 18,
                          ),
                        ),
                      ),
                      Container(
                        width:375,
                        height: 450,
                        child: Image.network(
                          _posts[index].imagePath,
                          fit: BoxFit.cover,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        _posts[index].description,
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                );
              }
            },
        ),
      ),
      bottomNavigationBar: BottomAppBar(
        color: Colors.blue, // Color de fondo azul
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            IconButton(
              icon: Icon(Icons.home,   color: Colors.yellow),
              onPressed: () {},
            ),
            IconButton(
              icon: Icon(Icons.play_circle,  color: Colors.yellow),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => CameraScreen(cameras: widget.cameras, userId: userId),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingIndicator() {
    return Center(
      child: CircularProgressIndicator(),
    );
  }

  Widget _buildLoadMoreButton() {
    return ElevatedButton(
      onPressed: _fetchPosts,
      child: Text('Cargar más'),
    );
  }
}