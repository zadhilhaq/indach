import 'dart:io';

import 'package:clipboard/clipboard.dart';
import 'package:indach/api/firebase_ml_api.dart';
import 'package:indach/widget/text_area_widget.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:developer';

import 'controls_widget.dart';

class TextRecognitionWidget extends StatefulWidget {
  const TextRecognitionWidget({
    Key key,
  }) : super(key: key);

  @override
  _TextRecognitionWidgetState createState() => _TextRecognitionWidgetState();
}

class _TextRecognitionWidgetState extends State<TextRecognitionWidget> {
  String text = '';
  File image;
  String final_response = "";
  bool upload = false;

  final url = 'http://192.168.43.47:5000/text';

  @override
  Widget build(BuildContext context) => Expanded(
        child: Column(
          children: [
            Expanded(child: buildImage()),
            const SizedBox(height: 16),
            text != ''
                ? ControlsWidget(
                    onClickedPickImage: pickImage,
                    onClickedScanText: scanText,
                    onClickedTranslation: translation,
                    onClickedCopy: copyToClipboard,
                    onClickedClear: clear,
                  )
                : image != null
                    ? ControlsWidget(
                        onClickedPickImage: pickImage,
                        onClickedScanText: scanText,
                      )
                    : ControlsWidget(
                        onClickedPickImage: pickImage,
                      ),
            const SizedBox(height: 16),
            TextAreaWidget(
              text: text,
            ),
          ],
        ),
      );

  Widget buildImage() => Container(
        child: image != null
            ? Image.file(image)
            : Icon(Icons.photo, size: 80, color: Colors.deepOrange),
      );

  Future pickImage() async {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Pilih Gambar'),
        content: Container(
          width: 70.0,
          height: 120.0,
          child: Column(
            children: [
              SimpleDialogOption(
                child: Text('Ambil dengan kamera'),
                onPressed: pickCamera,
              ),
              SimpleDialogOption(
                child: Text('Ambil dari Galeri'),
                onPressed: pickGaleri,
              ),
              SimpleDialogOption(
                child: Text('Batal'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              )
            ],
          ),
        ),
      ),
    );
  }

  Future pickGaleri() async {
    Navigator.of(context).pop();
    final file = await ImagePicker().getImage(source: ImageSource.gallery);
    setImage(File(file.path));
  }

  Future pickCamera() async {
    Navigator.of(context).pop();
    final file = await ImagePicker().getImage(source: ImageSource.camera);
    setImage(File(file.path));
  }

  Future scanText() async {
    showDialog(
        context: context, builder: (context) => CircularProgressIndicator());

    final text = await FirebaseMLApi.recogniseText(image);
    setText(text);
    Navigator.of(context).pop();
  }

  Future translation() async {
    showDialog(
        context: context, builder: (context) => CircularProgressIndicator());
    /*final text = await Translation.translate();
     setText(text);
  
      Navigator.of(context).pop();
      */
    await http
        .post(url,

            // "https://jsonplaceholder.typicode.com/posts"
            headers: {"Content-Type": "application/json", "charset": "UTF-8"},
            body: json.encode({"text": text}))
        .then((response) {
      var res = json.decode(response.body);
      log("Response awawawaw : $res");
      final decoded = json.decode(response.body) as Map<String, dynamic>;
      final text = (decoded["response"]);
      setText(text);

      return text;
    });
    Navigator.of(context).pop();
  }

// untuk hapus teks dan gambar
  void clear() {
    setImage(null);
    setText('');
  }

// untuk kopi teks
  void copyToClipboard() {
    if (text.trim() != '') {
      FlutterClipboard.copy(text);
    }
  }

  void setImage(File newImage) {
    setState(() {
      image = newImage;
    });
  }

  void setText(String newText) {
    setState(() {
      text = newText;
    });
  }
}
