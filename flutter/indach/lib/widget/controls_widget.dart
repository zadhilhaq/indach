import 'package:flutter/material.dart';
import '../ocricon_icons.dart';

class ControlsWidget extends StatelessWidget {
  final VoidCallback onClickedPickImage;
  final VoidCallback onClickedScanText;
  final VoidCallback onClickedTranslation;
  final VoidCallback onClickedClear;
  final VoidCallback onClickedCopy;

  const ControlsWidget({
    @required this.onClickedPickImage,
    this.onClickedScanText,
    this.onClickedTranslation,
    this.onClickedClear,
    this.onClickedCopy,
    Key key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) => new Container(
    child: new Center(
      child: new Column(
        children: <Widget>[
          new Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
            icon: const Icon(Icons.add_a_photo),
            onPressed: onClickedPickImage,
          ),
          const SizedBox(width: 4),
          IconButton(
            icon: const Icon(IconData(0xe800, fontFamily: 'Ocricon')),
            onPressed: onClickedScanText,
          ),
          const SizedBox(width: 4),
          /* onClickedScanText == null
            ? Container()
            : IconButton(icon: const Icon(
              IconData(0xe800, fontFamily:  'Ocricon')),
             onPressed: onClickedScanText),*/
          IconButton(
            icon: const Icon(Icons.translate),
            onPressed: onClickedTranslation,
          ),
          const SizedBox(width: 1),
          IconButton(
            icon: Icon(Icons.content_copy),
            onPressed: onClickedCopy,
          ),
          const SizedBox(width: 4),
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: onClickedClear,
          ),
            ],
          ),
        ],),
    ),
  );
}