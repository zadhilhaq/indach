import 'package:flutter/material.dart';

class TextAreaWidget extends StatelessWidget {
  final String text;
  

  const TextAreaWidget({
    @required this.text,
    Key key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) => Row(
        children: [
          
          SingleChildScrollView(
            child: Container(
              height: 200,
              width: MediaQuery.of(context).size.width*0.9,
              decoration: BoxDecoration(border: Border.all()),
              padding: EdgeInsets.all(8.0),
              alignment: Alignment.center,

              
              child: SelectableText(
                
                text.isEmpty ? 'Silahkan masukkan gambar untuk diterjemahkan!' : text,
                //textDirection: TextDirection.ltr,
                textAlign: TextAlign.center,

              ),
            ),
          ),
          
        ],
      );
}