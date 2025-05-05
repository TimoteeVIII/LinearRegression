from pydantic import BaseModel


class FeatureVector(BaseModel):
    crim: float | None
    zn: float | None
    indus: float | None
    chas: float | None
    nox: float | None
    rm: float | None
    age: float | None
    dis: float | None
    rad: float | None
    tax: float | None
    ptratio: float | None
    b: float | None
    lstat: float | None
