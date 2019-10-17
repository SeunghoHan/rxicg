from flask import Flask, request, render_template
#from flask_restful import Resource, Api
import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
ROOT_PATH = '/archive/MyHome/Programs/git/my_research/Image_Captioning/Demo/'
sys.path.insert(0, ROOT_PATH)

import demo

app = Flask(__name__, static_url_path = "", static_folder = "images")
#app = Flask(__name__, static_url_path = "", static_folder = "images/output/regions/")


def getTemplate():
    html = '<!DOCTYPE html>' \
           '<html>' \
           '<head>' \
           '<link href="http://bootstrapk.com/dist/css/bootstrap.min.css" rel="stylesheet">' \
           '<link href="http://bootstrapk.com/dist/css/bootstrap-theme.min.css" data-href="http://bootstrapk.com/dist/css/bootstrap-theme.min.css" rel="stylesheet" id="bs-theme-stylesheet">' \
           '</head>' \
           '<body>'


    html += '<nav class="navbar navbar-inverse navbar-fixed-top>' \
            '<div class="container">' \
            '<div class="navbar-header">' \
            '<a class="navbar-brand" href="/"><i>[Demo] Explainable Image Caption Generator</i></a>' \
            '</div>"' \
            '<div id="navbar" class="navbar-collapse collapse">' \
            '<ul class="nav navbar-nav">' \
            '<!--<li class="active"><a href="#">menu 1</a></li>-->' \
            '</ul></div>' \
            '</nav>'

    # Main body will be replaced
    html += '<div class="container">' \
            '{}' \
            '</div>'

    html += '</div></body>' \
            '</html>'

    return html

@app.route('/',)
def main():
    #img_path = '/archive/MyHome/Programs/git/my_research/Image Captioning/Demo/flaskapp/app/images/'
    img_path = 'images/'
    img_ids = []
    for file in os.listdir(img_path):
        if len(file.split('.')[0]) < 3:
            img_ids.append(file)


    html = getTemplate()

    cont = '<h2 style="margin:4px;margin-left:12px;"><b>Select a Test Image:</b></h2>'
    for idx in range(len(img_ids)):
        tmp_a = '<a href="/get_image?name=' + img_ids[idx] +'">{}</a>'
        cont += '<div class="col-xs-6 col-md-4" style="padding:4px">'
        cont += tmp_a.format('<img class="img-thumbnail" src="/' + img_ids[idx] + '" style="height:200px;width:100%;object-fit:cover"/>')
        cont += tmp_a.format('<center><h4>' + img_ids[idx] + '<h4></center>')
        cont += '</div>'

    return html.format(cont)

def _get_main_style():
    html = '<head><style>'
    html += 'body {margin:15px;}'
    html += 'ul {list-style-type: none; margin: 0; padding: 0; overflow: hidden; background-color: #333;}'
    html += 'li a {display: block; color: white; text-align: center; padding: 8px 8px; text-decoration: none; font-size: 24px;}'
    html += 'hr {display: block; margin-top: 0.5em; margin-bottom: 0.5em; margin-left: auto; margin-right: auto; border-style: inset; border-width: 5px;}'
    html += '</style></head>'
    return html


@app.route('/get_image',  methods=['GET'])
def get_image():
    img_name = request.args['name']

    region_path     = 'images/output/regions/'
    # execute testing for selected image
    demoTest = demo.DemoTest(img_name.split('.')[0])
    result = demoTest.test()
    # create the final output image and captions (as an img format)
    image_shape = demoTest.show_final_output(result)
    # create the region information and weight matrix
    output = demoTest.show_cap_and_wmatrix(result)

    html = getTemplate()

    cont = '<h2 style="margin:4px;margin-left:12px;"><b>Result for Selected Image</b></h2>'
    cont = ''

    cont += '<div class="panel panel-primary">' \
            '<div class="panel-heading">' \
            '<h3 class="panel-title" id="panel-title" style="font-size:24px">' \
            '<b>Final Output</b>' \
            '<a class="anchorjs-link" href="#panel-title"><span class="anchorjs-icon"></span></a></h3>' \
            '</div>' \
            '<div class="panel-body">' \
            '<h4><b>&#9654; Image ID: </b>' + result.image_id + '</h4>' \
            '<h4><b>&#9654; Image shape:  </b>' + str(image_shape) + '</h4>' \
            '<hr>' \
            '<center>' \
            '<img src="output/output_img.jpg" align="middle" style="max-width:100%"/>' \
            '<img src="output/output_cap.jpg" align="middle" style="max-width:100%"/>' \
            '</center>' \
            '</div>' \
            '</div>'

    cont += '<div class="panel panel-primary">' \
            '<div class="panel-heading">' \
            '<h3 class="panel-title" id="panel-title" style="font-size:24px">' \
            '<b>Weight Matrix Based on Region-Word Embedding</b>' \
            '<a class="anchorjs-link" href="#panel-title"><span class="anchorjs-icon"></span></a></h3>' \
            '</div>'


    cont += '<div class="panel-body">' \
            '<h4><b>&#9654; Region Information:</b></h4>' \
            '<center>'

    region_idx = 0
    for region in os.listdir(region_path):
        cont += '<div style="width:fit-content;display:inline-block;margin-left:4px;margin-right:4px">' \
                '<center><img src="output/regions/' + region + '"></center>' \
                '<div class="caption" style="margin:4px;margin-top:12px;margin-bottom:20px">' \
                '<ul style="text-align:left;width:fit-content"><li>Region #: ' + str(region_idx) + '</li>' \
                '<li>Related word: ' + region + '</li>' \
                '<li>Region shape: ' + str(output[region]) + '</li>'\
                '</ul>'\
                '</div>' \
                '</div>'
        region_idx += 1

    cont += '</center>'

    cont += '<hr>' \
            '<h4><b>&#9654; Weight Matrix between Regions and Words:</b></h4>' \
            '<center>' \
            '<img src="output/output_weight.jpg" align="middle" style="width:100%"/>' \
            '</center>'

    cont += '</div>'

    return html.format(cont)

def _get_result_style():
    html = '<head><style>'
    html += 'body {margin:15px;}'

    html += 'ul {list-style-type: none; margin: 0; padding: 0; overflow: hidden; background-color: #333;}'
    html += 'li a {display: block; color: white; text-align: center; padding: 8px 8px; text-decoration: none; font-size: 24px;}'

    html += 'hr {display: block; margin-top: 0.5em; margin-bottom: 0.5em; margin-left: auto; margin-right: auto; border-style: inset; border-width: 5px;}'
    html += 'mark {background-color: black; color: yellow;}'

    html += 'div.gallery {margin: 5px; float: center; width: 200px;}'
    html += 'div.gallery:hover {border: 1px solid #777;}'
    html += 'div.gallery img {width: auto; height: auto;}'
    html += 'div.desc {padding: 16px; text-align: center;}'
    html += '.region-size {width: auto; height: auto; max-width: 50%; max-height: 50%;}'
    html += '</style></head>'
    return html


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(debug=True)
