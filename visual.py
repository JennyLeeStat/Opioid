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

