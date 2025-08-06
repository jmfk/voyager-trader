"""Tests for base model infrastructure."""

from datetime import datetime
from typing import Any, List

import pytest

from voyager_trader.models.base import (
    AggregateRoot,
    BaseEntity,
    DomainEvent,
    Specification,
    ValueObject,
)


class TestValueObject(ValueObject):
    """Test value object implementation."""

    name: str
    value: int


class TestEntity(BaseEntity):
    """Test entity implementation."""

    name: str
    description: str


class TestAggregateRoot(AggregateRoot):
    """Test aggregate root implementation."""

    name: str
    items: List[str] = []

    def __init__(self, **data):
        super().__init__(**data)
        self._domain_events: List[DomainEvent] = []

    def is_valid(self) -> bool:
        return len(self.name) > 0

    def get_domain_events(self) -> List[DomainEvent]:
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events


class TestDomainEvent(DomainEvent):
    """Test domain event implementation."""

    test_data: str


class TestSpecification(Specification):
    """Test specification implementation."""

    def __init__(self, value: int):
        self.value = value

    def is_satisfied_by(self, entity: Any) -> bool:
        return hasattr(entity, "value") and entity.value > self.value


class TestVoyagerBaseModel:
    """Test VoyagerBaseModel functionality."""

    def test_model_creation(self):
        """Test basic model creation."""
        vo = TestValueObject(name="test", value=42)
        assert vo.name == "test"
        assert vo.value == 42

    def test_model_immutability(self):
        """Test that models are immutable."""
        vo = TestValueObject(name="test", value=42)
        with pytest.raises(ValueError):
            vo.name = "changed"

    def test_model_validation(self):
        """Test model validation."""
        with pytest.raises(ValueError):
            TestValueObject(name="test", value="invalid")  # Should be int


class TestBaseEntity:
    """Test BaseEntity base class."""

    def test_entity_creation(self):
        """Test entity creation with auto-generated fields."""
        entity = TestEntity(name="test", description="desc")

        assert entity.name == "test"
        assert entity.description == "desc"
        assert entity.id is not None
        assert len(entity.id) > 0
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)
        assert entity.version == 1

    def test_entity_equality(self):
        """Test entity equality based on ID."""
        entity1 = TestEntity(name="test1", description="desc1")
        entity2 = TestEntity(name="test2", description="desc2")
        entity3 = TestEntity(id=entity1.id, name="test3", description="desc3")

        assert entity1 != entity2  # Different IDs
        assert entity1 == entity3  # Same ID
        assert entity1 != "not an entity"

    def test_entity_hash(self):
        """Test entity hashing based on ID."""
        entity1 = TestEntity(name="test1", description="desc1")
        entity2 = TestEntity(id=entity1.id, name="test2", description="desc2")

        assert hash(entity1) == hash(entity2)

        # Can be used in sets
        entity_set = {entity1, entity2}
        assert len(entity_set) == 1  # Same ID

    def test_entity_update(self):
        """Test entity update mechanism."""
        entity = TestEntity(name="original", description="desc")
        original_created = entity.created_at
        original_version = entity.version

        updated = entity.update(name="updated")

        assert updated.name == "updated"
        assert updated.description == "desc"  # Unchanged
        assert updated.id == entity.id  # Same ID
        assert updated.created_at == original_created  # Unchanged
        assert updated.updated_at > entity.updated_at  # Updated
        assert updated.version == original_version + 1  # Incremented


class TestSpecificationPattern:
    """Test Specification pattern implementation."""

    def test_specification_evaluation(self):
        """Test specification evaluation."""
        spec = TestSpecification(10)

        # Mock objects for testing
        high_value = TestValueObject(name="high", value=20)
        low_value = TestValueObject(name="low", value=5)

        assert spec.is_satisfied_by(high_value)
        assert not spec.is_satisfied_by(low_value)

    def test_and_specification(self):
        """Test AND specification combination."""
        spec1 = TestSpecification(5)
        spec2 = TestSpecification(10)
        and_spec = spec1.and_(spec2)

        high_value = TestValueObject(name="high", value=20)
        mid_value = TestValueObject(name="mid", value=8)
        low_value = TestValueObject(name="low", value=3)

        assert and_spec.is_satisfied_by(high_value)  # > 5 AND > 10
        assert not and_spec.is_satisfied_by(mid_value)  # > 5 but not > 10
        assert not and_spec.is_satisfied_by(low_value)  # not > 5 and not > 10

    def test_or_specification(self):
        """Test OR specification combination."""
        spec1 = TestSpecification(15)
        spec2 = TestSpecification(5)
        or_spec = spec1.or_(spec2)

        high_value = TestValueObject(name="high", value=20)
        mid_value = TestValueObject(name="mid", value=8)
        low_value = TestValueObject(name="low", value=3)

        assert or_spec.is_satisfied_by(high_value)  # > 15 OR > 5
        assert or_spec.is_satisfied_by(mid_value)  # not > 15 but > 5
        assert not or_spec.is_satisfied_by(low_value)  # not > 15 and not > 5

    def test_not_specification(self):
        """Test NOT specification."""
        spec = TestSpecification(10)
        not_spec = spec.not_()

        high_value = TestValueObject(name="high", value=20)
        low_value = TestValueObject(name="low", value=5)

        assert not not_spec.is_satisfied_by(high_value)  # NOT (> 10)
        assert not_spec.is_satisfied_by(low_value)  # NOT (> 10)

    def test_complex_specification_combination(self):
        """Test complex specification combinations."""
        spec1 = TestSpecification(5)
        spec2 = TestSpecification(15)
        spec3 = TestSpecification(10)

        # (> 5 AND > 15) OR NOT (> 10)
        complex_spec = spec1.and_(spec2).or_(spec3.not_())

        very_high = TestValueObject(name="very_high", value=20)  # Satisfies all
        high = TestValueObject(name="high", value=12)  # > 5, not > 15, > 10
        low = TestValueObject(name="low", value=3)  # not > 5, not > 15, not > 10

        assert complex_spec.is_satisfied_by(
            very_high
        )  # (True AND True) OR False = True
        assert not complex_spec.is_satisfied_by(
            high
        )  # (True AND False) OR False = False
        assert complex_spec.is_satisfied_by(low)  # (False AND False) OR True = True


if __name__ == "__main__":
    pytest.main([__file__])
