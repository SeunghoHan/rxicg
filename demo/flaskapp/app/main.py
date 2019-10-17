from flask import Flask 
app = Flask(__name__)

@app.route('/')
def main():
	return render_template('hello.html')
    #return 'Hello Webisfree!'

@app.route('/about/')
def about():
  return 'About page'


if __name__ == '__main__':
  app.run(debug=True)