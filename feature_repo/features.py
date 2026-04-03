from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from datetime import timedelta

iris_entity = Entity(
    name="entity_id",
    join_keys=["entity_id"],
    description="Iris sample entity"
)

iris_source = FileSource(
    path="processed_data/stock_data.parquet",
    event_timestamp_column="event_timestamp"
)

iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width",  dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width",  dtype=Float64),
        Field(name="sepal_ratio",  dtype=Float64),
        Field(name="petal_ratio",  dtype=Float64),
        Field(name="sepal_area",   dtype=Float64),
        Field(name="petal_area",   dtype=Float64),
        Field(name="species",      dtype=Int64),
    ],
    source=iris_source,
)
