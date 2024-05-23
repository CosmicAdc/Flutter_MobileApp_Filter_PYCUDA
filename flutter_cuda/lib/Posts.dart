class Post {
  final int id;
  final int userId;
  final String imagePath;
  final String description;

  Post({required this.id, required this.userId, required this.imagePath, required this.description});

  factory Post.fromJson(Map<String, dynamic> json) {
    String prepath=json['image_path'];
    String imagePath = prepath.substring(4);
    return Post(
      id: json['id'],
      userId: json['id_user'],
      imagePath: 'http://10.0.2.2:8000/'+imagePath,
      description: json['description'],
    );
  }


}
