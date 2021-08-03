import functools

from pydantic import BaseModel


@functools.total_ordering
class Model(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        self.__pre_init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.__post_init__()

    def _get_immutable_fields(self):
        return tuple((f, getattr(self, f)) for f in sorted(self.__fields__))

    def __hash__(self):
        fields = (type(self), *self._get_immutable_fields())
        return hash(fields)

    def __eq__(self, other):
        return self._get_immutable_fields() == other._get_immutable_fields()

    def __lt__(self, other):
        return self._get_immutable_fields() < other._get_immutable_fields()

    def __pre_init__(self, *args, **kwargs):
        pass

    def __post_init__(self):
        pass
