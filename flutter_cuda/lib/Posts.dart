import 'package:flutter_cuda/appConfig.dart';

class Post {
  final int id;
  final int userId;
  final String imagePath;
  final String description;
  final String username;

  Post({required this.id, required this.userId, required this.imagePath, required this.description,required this.username});

  factory Post.fromJson(Map<String, dynamic> json) {
    String prepath=json['image_path'];
    String imagePath = prepath.substring(4);
    return Post(
      id: json['id'],
      userId: json['id_user'],
      imagePath: '${AppConfig.apiUrl}:${AppConfig.port}/'+imagePath,
      description: json['description'],
      username: json['username'],
    );
  }


}
