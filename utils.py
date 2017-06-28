import pandas as pd
import os
import sys
import urllib
import logging
import zipfile

logging.basicConfig(
    format= '%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)



opioids = pd.read_csv("dataset/opioids.csv")
overdose = pd.read_csv("dataset/overdoses.csv", thousands = ',')
prescriber = pd.read_csv("dataset/prescriber-info.csv")


def download_and_decompress(url, dest_dir):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    filename = url.split('/')[ -1 ]
    filepath = os.path.join(dest_dir, filename)
    uncomp_filedir = filename.split('.')[ 0 ]
    #uncomp_filepath = os.path.join(dest_dir, uncomp_filedir)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.isfile(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath,
                                                 reporthook=_progress)
        statinfo = os.stat(filepath)
        logging.info('Successfully downloaded {}'.format(filename))
        logging.info('{} bytes.'.format(statinfo.st_size))

    if not os.path.isfile(filepath):
        logging.info("Uncompressing {}".format(filename))
        zipfile.ZipFile(filepath, 'r').extractall(dest_dir)
        logging.info(uncomp_filedir + ' successfully uncompressed')
        print()

    logging.info("Data set {}".format(filename))
    logging.info("from url: {}".format(url))
    logging.info("successfully downloaded and uncompressed")


def clean_txt(series):
    cleaned = series.str.lower().str.strip().str.replace('/', "_").str.replace('-', "_")
    cleaned = cleaned.str.replace(' ', '_').str.replace(',', '_').str.replace('__', '_')
    return cleaned


def plot_us_map(df, state, code, z,
                title='Overdose Deaths Per Capita',
                colorbar_title=None):

    scl = [ [ 0.0, 'rgb(242,240,247)' ],
            [ 0.2, 'rgb(218,218,235)' ],
            [ 0.4, 'rgb(188,189,220)' ],
            [ 0.6, 'rgb(158,154,200)' ],
            [ 0.8, 'rgb(117,107,177)' ],
            [ 1.0, 'rgb(84,39,143)' ] ]

    for col in df.columns:
        df[ col ] = df[ col ].astype(str)

    df[ 'text' ] = df[state] + '<br>' + z + df[z]

    data = [ dict(
        type='choropleth',
        colorscale=scl,
        autocolorscale=True,
        locations=df[code],
        z=df[z].astype(float),
        locationmode='USA-states',
        text=df[ 'text' ],
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
        colorbar = dict(
        title=colorbar_title)
    ) ]

    layout = dict(
        title=title,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=False,
            lakecolor='rgb(255, 255, 255)'),
    )

    fig = dict(data=data, layout=layout)
    return fig

















