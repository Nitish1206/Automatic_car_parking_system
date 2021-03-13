# from flask import Flask,jsonify
#
# app=Flask(__name__)
#
# @app.route("/")
# def send_data():
#     return jsonify({"id":"hello","color":"green","text":"this is text"})
#
# if __name__=="__main__":
#     app.run()

raw_data=[2,3,5,10]
print(raw_data)
raw_data[2]=8
print(raw_data)