import pytest
from pydantic import ValidationError

from patos import (
    Component,
    FlexModel,
    FrozenFlexModel,
    FrozenModel,
    FrozenOpenModel,
    InternedComponent,
    Model,
    OpenModel,
    Runtime,
)

# One real OIDC discovery document, trimmed to the shape that took a deploy down. A reader
# declares the endpoints it calls, and the provider advertises introspection, revocation and
# back-channel logout metadata beside them, which the spec allows and which the strict bases
# turned into a hard failure at startup.
_DISCOVERY = {
    "issuer": "https://auth.example.com/oidc",
    "jwks_uri": "https://auth.example.com/oidc/jwks",
    "token_endpoint": "https://auth.example.com/oidc/token",
    "introspection_endpoint": "https://auth.example.com/oidc/token/introspection",
    "revocation_endpoint": "https://auth.example.com/oidc/token/revocation",
    "backchannel_logout_supported": True,
    "backchannel_logout_session_supported": True,
    "claim_types_supported": ["normal"],
}


def test_model_is_mutable_and_standard() -> None:
    """`Model` mutates freely and rejects arbitrary types it cannot validate."""

    class Point(Model):
        x: int

    point = Point(x=1)
    point.x = 2
    assert point.x == 2


def test_frozen_model_is_frozen_and_validates_by_name() -> None:
    """`FrozenModel` declares the frozen, populate-by-name config its records rely on."""

    class Config(FrozenModel):
        size: int

    assert Config.model_config["frozen"] is True
    assert Config.model_config["extra"] == "forbid"
    assert Config.model_config["populate_by_name"] is True
    assert Config(size=8).size == 8
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config.model_validate({"size": 8, "sizes": 9})


def test_frozen_model_stable_id_is_content_derived_and_process_stable() -> None:
    """Stable identity includes model ownership and canonical validated content."""

    class Point(FrozenModel):
        x: int
        labels: dict[str, int]

    first = Point(x=1, labels={"beta": 2, "alpha": 1})
    reordered = Point(x=1, labels={"alpha": 1, "beta": 2})
    changed = Point(x=2, labels={"alpha": 1, "beta": 2})

    assert 0 <= first.stable_id < 2**64
    assert first.stable_id == reordered.stable_id
    assert first.stable_id != changed.stable_id


def test_open_model_keeps_declared_fields_and_tolerates_upstream_additions() -> None:
    """`OpenModel` reads what it declares out of a payload that advertises more."""

    class Reader(OpenModel):
        issuer: str
        token_endpoint: str

    reader = Reader.model_validate(_DISCOVERY)

    assert Reader.model_config["extra"] == "ignore"
    assert reader.issuer == _DISCOVERY["issuer"]
    assert reader.token_endpoint == _DISCOVERY["token_endpoint"]
    assert reader.model_extra is None
    assert reader.model_dump() == {
        "issuer": _DISCOVERY["issuer"],
        "token_endpoint": _DISCOVERY["token_endpoint"],
    }
    reader.issuer = "https://auth.example.com/oidc/"
    assert reader.issuer.endswith("/")


def test_frozen_open_model_tolerates_upstream_additions_and_stays_a_frozen_record() -> None:
    """`FrozenOpenModel` keeps every frozen guarantee while accepting undeclared fields."""

    class Discovery(FrozenOpenModel):
        issuer: str
        jwks_uri: str
        token_endpoint: str

    discovery = Discovery.model_validate(_DISCOVERY)

    assert Discovery.model_config["extra"] == "ignore"
    assert Discovery.model_config["frozen"] is True
    assert Discovery.model_config["populate_by_name"] is True
    assert isinstance(discovery, FrozenModel)
    assert discovery.jwks_uri == _DISCOVERY["jwks_uri"]
    assert discovery.model_extra is None


def test_frozen_open_model_stable_id_ignores_what_the_provider_added() -> None:
    """A provider advertising more metadata leaves the parsed record's identity unchanged."""

    class Discovery(FrozenOpenModel):
        issuer: str
        token_endpoint: str

    declared = {
        "issuer": _DISCOVERY["issuer"],
        "token_endpoint": _DISCOVERY["token_endpoint"],
    }

    assert Discovery.model_validate(declared).stable_id == (
        Discovery.model_validate(_DISCOVERY).stable_id
    )


def test_open_bases_do_not_loosen_the_strict_ones() -> None:
    """Opening one payload's schema never reaches the bases every authored payload uses."""

    class Authored(FrozenModel):
        size: int

    class Foreign(FrozenOpenModel):
        size: int

    assert Authored.model_config["extra"] == "forbid"
    assert Foreign.model_config["extra"] == "ignore"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Authored.model_validate({"size": 8, "sizes": 9})


def test_flex_model_accepts_arbitrary_types() -> None:
    """`FlexModel` carries a non-pydantic type that the strict bases would reject."""

    class Holder:
        pass

    class Box(FlexModel):
        item: Holder

    holder = Holder()
    assert Box(item=holder).item is holder


def test_frozen_flex_model_is_frozen_and_arbitrary() -> None:
    """`FrozenFlexModel` extends every strict frozen-model guarantee."""

    class Holder:
        pass

    class Box(FrozenFlexModel):
        item: Holder

    assert Box.model_config["frozen"] is True
    assert Box.model_config["arbitrary_types_allowed"] is True
    assert Box.model_config["extra"] == "forbid"
    assert Box.model_config["populate_by_name"] is True
    holder = Holder()
    box = Box(item=holder)
    assert box.item is holder
    assert isinstance(box, FrozenModel)


def test_runtime_marks_an_already_validated_boundary() -> None:
    """`Runtime` keeps any owned live value without asking Pydantic to schema its type."""

    class Service:
        pass

    class Box(FrozenModel):
        value: Runtime[int]
        service: Runtime[Service]

    service = Service()
    box = Box.model_validate({"value": "already checked", "service": service})

    assert box.value == "already checked"
    assert box.service is service
    assert Box.model_json_schema()["$defs"]["Runtime_Service_"] == {}


def test_component_self_registers_with_kebab_name() -> None:
    """A `Component` subclass enrolls in the registry under its kebab-cased class name."""

    class Codec(Component):
        bits: int

    class OpusFB8(Codec):
        pass

    assert OpusFB8.name == "opus-fb8"
    assert OpusFB8 in Codec.implementations()
    assert Codec.find("opus-fb8") is OpusFB8


def test_interned_component_interns_by_construction_arguments() -> None:
    """`InternedComponent` returns the same object for identical construction arguments.

    Interning keys on the call arguments, like `lru_cache`, so two no-arg builds share one
    instance and two `dim=16` builds share another, while distinct arguments stay distinct.
    """

    class Lattice(InternedComponent):
        dim: int = 8

    assert Lattice() is Lattice()
    assert Lattice(dim=16) is Lattice(dim=16)
    assert Lattice(dim=16) is not Lattice(dim=8)
