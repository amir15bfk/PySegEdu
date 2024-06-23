from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import dodge
from bokeh.io import output_notebook

def plot_metrics(data,name=''):
    """
    The function `plot_metrics` generates a bar plot displaying performance metrics for different
    datasets, with the option to specify a name for the model.
    
    :param data: The `data` parameter in the `plot_metrics` function is a dictionary containing the
    metrics data for plotting. It should have keys for 'Metrics', 'Kvasir', and 'CVC' with corresponding
    values for each metric and dataset. The 'Metrics' key should contain the list of metrics
    :param name: The `name` parameter in the `plot_metrics` function is used to specify the name of the
    model for which the performance metrics are being plotted. This name will be included in the title
    of the plot to provide context about which model the metrics belong to. If no `name` is provided,
    """
    
    source = ColumnDataSource(data=data)

    # Output to notebook
    output_notebook()
    # Create a new plot
    p = figure(x_range=data['Metrics'], y_range=(50, 100), height=400, title=f"{name} model performence",
            toolbar_location="above", tools="pan,wheel_zoom,box_zoom,reset,save")

    # Add bars for Kvasir and CVC datasets
    if data.get("Kvasir"):
        p.vbar(x=dodge('Metrics', -0.15, range=p.x_range), top='Kvasir', width=0.3, source=source, color="#718dbf", legend_label="Kvasir")
    if data.get("CVC"):
        p.vbar(x=dodge('Metrics', 0.15, range=p.x_range), top='CVC', width=0.3, source=source, color="#e84d60", legend_label="CVC")

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [ 
        ("Metric", "@Metrics"),
        ("Kvasir", "@Kvasir%"),
        ("CVC", "@CVC%")
    ]
    p.add_tools(hover)
    # Customize plot
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "bottom_left"
    p.legend.orientation = "horizontal"
    p.yaxis.axis_label = "Percentage (%)"

    
    # Show the plot

    show(p)