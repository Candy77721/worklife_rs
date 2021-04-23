import json
from flask import Flask, redirect, url_for, request, render_template

# %%
'''
app = Flask(__name__)

@app.route('/', methods=['GET'])  # http://10.100.11.19:8000/
def adddemo():
    return render_template('flask.html')

@app.route('/ww', methods=['GET']) # http://10.100.11.19:8000/ww
def index():  
    return "欢迎访问77的网站"
'''
# %%
app = Flask(__name__)

@app.route('/main')  # http://10.100.11.19:8000/main
def index():
    return render_template('flask.html')

@app.route('/getformdata/<name>') # http://10.100.11.19:8000/getformdata/123
def success(name):
    return 'welcome %s' % name

@app.route('/login_html',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success',name = user)) # url_for操作对象是函数，而不是route里的路径。
    else:
        user = request.args.get('nm')
        return redirect(url_for('success',name = user))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
