import 'package:indach/widget/text_recognition_widget.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  final String title = 'INDACH';

  @override
  Widget build(BuildContext context)
  => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: title,
        theme: ThemeData(primarySwatch: Colors.deepOrange),
        home: MainPage(title: title),
      );     
}

class MainPage extends StatefulWidget {
  final String title;

  const MainPage({
    @required this.title,
  });

  @override
  _MainPageState createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: Center(
            child: Text(widget.title)
          ),
          
          leading: Image.asset(
            'assets/logo.png',
            ),
           
          
        ),
        body: Padding(
          padding: const EdgeInsets.all(8),
          child: Column(
            children: [
              const SizedBox(height: 25),
              TextRecognitionWidget(),
              const SizedBox(height: 15),
            ],
          ),
        ),
      );
}

