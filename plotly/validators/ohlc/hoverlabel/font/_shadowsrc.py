import _plotly_utils.basevalidators


class ShadowsrcValidator(_plotly_utils.basevalidators.SrcValidator):
    def __init__(
        self, plotly_name="shadowsrc", parent_name="ohlc.hoverlabel.font", **kwargs
    ):
        super(ShadowsrcValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            edit_type=kwargs.pop("edit_type", "none"),
            **kwargs,
        )
