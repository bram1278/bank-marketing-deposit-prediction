from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home_recall.html')

@app.route('/visual')
def visual():
    df = pd.read_csv('dataset/bank-additional-full.csv', index_col =0)

    # Menentukan Size
    j = []
    for i in df["poutcome"].unique():
        j.append(df[(df["poutcome"]==i) & (df['y'] == 'yes')]["poutcome"].count())

    persentase = []
    nama = []
    for i in range(0,(len(j))):
        persentase.append(j[i])
        nama.append(df["poutcome"].unique()[i])

    x, y, z = plt.pie(
        persentase,
        labels=nama,    
        startangle=0,
        shadow=True,
        radius = 1,
        textprops={
        'size' : 15 ,
        'color' : 'black'
        },

        autopct = '%1.1f%%',
        explode = (.05, .05, .05) # pemisah pie chart
    )

    for i in z:
        i.set_color('white')    
    
    plt.savefig('pie_chart_poutcome.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.catplot(x='y', y='duration', data = df, kind = 'bar', aspect = 1)
        
    plt.savefig('barplot_duration.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result2 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.catplot(x='y', y='pdays', data = df, kind = 'bar', aspect = 1)

    plt.savefig('barplot_pdays.png',bbox_inches="tight") 

    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result3 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.catplot(x='y', y='previous', data = df, kind = 'bar', aspect = 1)
    
    plt.savefig('barplot_previous.png',bbox_inches="tight") 

    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result4 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.catplot(x='y', y='emp.var.rate', data = df, kind = 'bar', aspect = 1)
    
    plt.savefig('barplot_employment_variation_rate.png',bbox_inches="tight") 

    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result5 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.catplot(x='y', y='euribor3m', data = df, kind = 'bar', aspect = 1)
    
    plt.savefig('barplot_euribor3m.png',bbox_inches="tight") 

    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result6 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    sns.catplot(x='y', y='nr.employed', data = df, kind = 'bar', aspect = 1)
    
    plt.savefig('barplot_nr_employed.png',bbox_inches="tight") 

    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result7 = str(figdata_png)[2:-1]

    return render_template('plot.html', plot = result, plot2 = result2, plot3 = result3, plot4 = result4, plot5 = result5, plot6 = result6, plot7 = result7)

@app.route('/bank_recall')
def bank():
    return render_template('bank_recall.html')

@app.route('/klasifikasi_recall', methods = ['POST', 'GET'])
def hasil():
    if request.method == 'POST':
        input = request.form
        duration = float(input['duration'])
        pdays = float(input['pdays'])
        previous = float(input['previous'])
        emp_var_rate = float(input['emp_var_rate'])
        euribor3m = float(input['euribor3m'])
        nr_employed = float(input['nr_employed'])
        poutcome = float(input['poutcome'])
        if poutcome == 0:
            poutcome_failure = 1
            poutcome_nonexistent = 0
            poutcome_success = 0
        elif poutcome == 1:
            poutcome_failure = 0
            poutcome_nonexistent = 1
            poutcome_success = 0
        elif poutcome == 2:
            poutcome_failure = 0
            poutcome_nonexistent = 0
            poutcome_success = 1

        pred = Model.predict([[duration,pdays,previous,emp_var_rate,euribor3m,nr_employed,poutcome_failure,poutcome_nonexistent,poutcome_success]])[0]

        return render_template('hasil_recall.html', data=input, prediksi=pred)

if __name__ == "__main__":
    with open('recall_biased_model', 'rb') as model:
        Model = pickle.load(model)
    app.run(debug=True)













