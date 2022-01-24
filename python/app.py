from flask import Flask, jsonify, request
import json
# impor sent_tokenize dari modul nltk
import nltk
from nltk.tokenize import sent_tokenize
# import transliterate module to pass text for translation and get the translation result
from transliterate import translate as tr

#declared an empty variable for reassignment
response = ''

#creating the instance of our flask application
app = Flask(__name__)

#route to entertain our post and get request from flutter app
@app.route('/text', methods = ['GET', 'POST'])
def nameRoute():

    #fetching the global response variable to manipulate inside the function
    global response

    #checking the request type we get from the app
    if(request.method == 'POST'):
        request_data = request.data #getting the response data
        request_data = json.loads(request_data.decode('utf-8')) 
        text = request_data['text']
        myArr = sent_tokenize(text) 
        print(myArr)
        myArr2 = []
        for i in range(len(myArr)):
            
            

        #should call transliterate.translate() here
            translated = tr(myArr[i])
            responses = f'{translated}'
            s1 = responses.replace(' . <end>', '')
            s2 = s1.replace("['",'')
            s3 = s2.replace("']",'')
            myArr2.append(s3)

        #response = f'hasil terjemahan : {text}!' #re-assigning response with the name we got from the user
        #response = transliterate.translate(text_input, translation_output)
        # return "aw" #to avoid a type error 
        x = ". ".join(myArr2)
        response = x
        return jsonify({'response' : response})
    else:
        return jsonify({'text' : "response"}) #sending data back to your frontend app

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='192.168.43.47')
