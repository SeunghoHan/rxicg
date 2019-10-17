from flask import Flask, request, render_template
#from flask_restful import Resource, Api
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import demo

app = Flask(__name__, static_url_path = "", static_folder = "images")
#app = Flask(__name__, static_url_path = "", static_folder = "images/output/regions/")


@app.route('/',)
def main():
    #img_path = '/archive/MyHome/Programs/git/my_research/Image Captioning/Demo/flaskapp/app/images/'
    img_path = 'images/'
    img_ids = []
    for file in os.listdir(img_path):
        if len(file.split('.')[0]) < 3:
            img_ids.append(file)

    html = _get_main_style()
    html += '<body>'
    html += '<ul> <li><a class="active"><b>[Demo] Explainable Image Caption Generator</b></a></li> </ul>'
    html += '<link href="http://bootstrapk.com/dist/css/bootstrap.min.css" rel="stylesheet">'
    html += '<link href="http://bootstrapk.com/dist/css/bootstrap-theme.min.css" data-href="http://bootstrapk.com/dist/css/bootstrap-theme.min.css" rel="stylesheet" id="bs-theme-stylesheet">'

    html += '<h2> <b>Image selection for testing</b> </h2><hr>'

    html += '<center>'
    for idx in range(len(img_ids)):
        html += '<a href="/get_image?name=' + img_ids[idx] +'"><img src="/' + img_ids[idx] + '" class="img-thumbnail" style="width:30%;height:30%;object-fit:cover;margin:6px"/></a>'
    html += '</center>'
    html += '</body>'

    return html

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

    region_path = 'images/output/regions/'
    # execute testing for selected image
    demoTest = demo.DemoTest(img_name.split('.')[0])
    result = demoTest.test()
    # create the final output image and captions (as an img format)
    demoTest.show_final_output(result)
    # create the region information and weight matrix
    output = demoTest.show_cap_and_wmatrix(result)


    html = _get_result_style()
    html += '<body>'
    html += '<ul> <li><a class="active"><b>[Demo] Explainable Image Caption Generator</b></a></li> </ul>'

    html += '<link href="http://bootstrapk.com/dist/css/bootstrap.min.css" rel="stylesheet">'
    html += '<link href="http://bootstrapk.com/dist/css/bootstrap-theme.min.css" data-href="http://bootstrapk.com/dist/css/bootstrap-theme.min.css" rel="stylesheet" id="bs-theme-stylesheet">'

    html += '<h2><b>Result for selected image</b></h2><hr>'
    html += '<h3><mark>Final output</mark></h3>'
    html += '<h4>Image ID: ' + result.image_id + '</h4>'
    html += '<h4>Processing 1 image </h4><br>'
    html += '<img src="output/output_img.jpg" align="middle"/><br>'
    html += '<img src="output/output_cap.jpg" align="middle"/><hr>'


    html += '<h3><mark>Weight matrix based on region-word embedding</mark></h3>'
    html += '<h4><b>Region information</b></h4>'
    region_idx = 0
    for region in os.listdir(region_path):
        html += '<div class="gallery">'
        r_width = output[region][0]
        r_height = output[region][1]
        if r_width > 150 or r_height > 150:
            html += '<img class="region-size" src="output/regions/' + region + '">'
        else:
            html += '<img src="output/regions/' + region + '">'
        html += '<div calss="desc">Region #: ' + str(region_idx) + '<br>' \
                + 'Related word: ' + region + '<br>' + 'Region shape' + str(output[region]) + '</div></div> <br>'
        # html += '<h4>Region ' + str(region_idx) + '</h4>'
        # html += '<h4>Related word: ' + region + '</h4>'
        # html += '<h4>Region shape: ' + str(output[region]) + '</h4>'
        region_idx += 1

    html += '<br><hr>'
    html += '<h4><b>Weight matrix between regions and words</b></h4>'
    html += '<img src="output/output_weight.jpg" align="middle"/><br>'
    html += '</body>'

    return html

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
