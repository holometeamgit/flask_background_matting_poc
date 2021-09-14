import os

import flask
from flask import Flask, stream_with_context
from flask import render_template, request

from inference_video_yield import process

app = Flask(__name__, static_url_path='/static')


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Welcome to </br>AWESOME BEEM SERVER!</h1>"


@app.route('/upload')
def upload_file_page():
    return render_template('upload.html')


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        project_name = request.form['project']
        project_dir = os.path.join("content_projects", project_name)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        f1 = request.files['video_file']
        complete_name1 = os.path.join(project_dir, "input.mp4")
        f1.save(complete_name1)

        f2 = request.files['bg_file']
        complete_name2 = os.path.join(project_dir, "input.jpg")
        f2.save(complete_name2)

        output_dir_var = os.path.join(os.path.join("static", "content_projects_output"), project_name)
        server_uri_var = request.base_url[:-8]

        result = process(video_src=complete_name1,
                         video_bgr=complete_name2,
                         output_dir=output_dir_var,
                         server_uri=server_uri_var)

        return flask.Response(stream_with_context(stream_template('template.html', rows=result)))

        # return flask.Response(process(video_src=complete_name1,
        #                               video_bgr=complete_name2,
        #                               output_dir=output_dir_var,
        #                               server_uri=server_uri_var))


@app.route('/template_test')
def template_test():
    result = ['1/10', '2/10', '3/10', '4/10', '5/10', '6/10']
    return render_template('template.html', rows=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=True)
