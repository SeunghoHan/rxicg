# -*- coding: utf-8 -*-

from untitled4_TIE_byRules import getTimex3
from datetime import datetime

from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/time', methods=['GET', 'POST'])
def time():
    timeStarted = datetime.now()
    
    if request.method == 'GET':
        input_data = request.args['s']
        print('[%s] input_data= %s' % (datetime.now().isoformat(' '), input_data))
        
        timex3s = getTimex3(input_data)
        strTimex3s = json.dumps(timex3s, sort_keys=True, indent=4, ensure_ascii=False)
        print('[%s]   > TIMEX3 tags= %s' % (datetime.now().isoformat(' '), strTimex3s))
        
        elapsedTime = datetime.now() - timeStarted
        outRes = '> input_data= "%s"\n> TIMEX3 tags= %s\n> Elapsed time=%s' % (input_data, strTimex3s, elapsedTime)
        return '<pre>%s</pre>' % outRes.replace('<', '&lt').replace('>', '&gt')
    
        pass
    elif request.method == 'POST':
        pass
#        input_data = request.form
#        print(input_data)
#        
#        ###### 시간정보추출 연동
#        r = requests.post("http://143.248.55.244/textAnalysis2.php", data=input_data)
#        print(r.status_code, r.reason)
#        
#        result_spotl = r.content.decode('utf-8')
#        print(result_spotl[:300] + '...')
#        ######
#    
#        return result_spotl

    return ''


if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=25025)
    