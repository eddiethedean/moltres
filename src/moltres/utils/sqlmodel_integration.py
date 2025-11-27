"""SQLModel and Pydantic integration utilities for Moltres."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

if TYPE_CHECKING:
    pass  # Type hints only


def is_sqlmodel_model(obj: Any) -> bool:
    """Detect if an object is a SQLModel model class.

    Args:
        obj: Object to check

    Returns:
        True if obj is a SQLModel model class, False otherwise
    """
    try:
        from sqlmodel import SQLModel

        # Check if it's a class
        if not isinstance(obj, type):
            return False

        # Check if it's a subclass of SQLModel
        if issubclass(obj, SQLModel):
            # Check if it has table=True (SQLModel table models)
            # SQLModel models with table=True have __table__ attribute
            if hasattr(obj, "__table__"):
                return True
            # Also check for __tablename__ which SQLModel models can have
            if hasattr(obj, "__tablename__"):
                return True

        return False
    except ImportError:
        return False


def is_pydantic_model(obj: Any) -> bool:
    """Detect if an object is a Pydantic BaseModel class.

    Args:
        obj: Object to check

    Returns:
        True if obj is a Pydantic BaseModel class, False otherwise
    """
    try:
        from pydantic import BaseModel

        # Check if it's a class
        if not isinstance(obj, type):
            return False

        # Check if it's a subclass of BaseModel
        # Exclude SQLModel since that's handled separately
        if issubclass(obj, BaseModel):
            # Don't treat SQLModel as pure Pydantic
            try:
                from sqlmodel import SQLModel

                if issubclass(obj, SQLModel):
                    return False
            except ImportError:
                pass
            return True

        return False
    except ImportError:
        return False


def is_model_class(obj: Any) -> bool:
    """Detect if an object is a SQLModel or Pydantic model class.

    Args:
        obj: Object to check

    Returns:
        True if obj is a SQLModel or Pydantic model class, False otherwise
    """
    return is_sqlmodel_model(obj) or is_pydantic_model(obj)


def get_sqlmodel_table_name(model_class: Type) -> str:
    """Extract table name from a SQLModel model class.

    Args:
        model_class: SQLModel model class

    Returns:
        Table name

    Raises:
        ValueError: If model doesn't have a table name
    """
    if hasattr(model_class, "__tablename__"):
        tablename = getattr(model_class, "__tablename__")
        if isinstance(tablename, str):
            return tablename
    if hasattr(model_class, "__table__"):
        # Fallback to __table__.name
        table = getattr(model_class, "__table__")
        if table is not None and hasattr(table, "name"):
            return str(table.name)
    # Fallback to class name (lowercased)
    return model_class.__name__.lower()


def model_to_dict(instance: Any) -> Dict[str, Any]:
    """Convert a SQLModel or Pydantic instance to a dictionary.

    Args:
        instance: SQLModel or Pydantic instance

    Returns:
        Dictionary representation of the instance

    Raises:
        TypeError: If instance is not a SQLModel or Pydantic instance
    """
    # Try SQLModel first
    try:
        from sqlmodel import SQLModel

        if isinstance(instance, SQLModel):
            # SQLModel instances have model_dump() method (Pydantic v2) or dict() method (Pydantic v1)
            if hasattr(instance, "model_dump"):
                return instance.model_dump()
            elif hasattr(instance, "dict"):
                return instance.dict()
            else:
                # Fallback to __dict__
                return {k: v for k, v in instance.__dict__.items() if not k.startswith("_")}
    except ImportError:
        pass

    # Try Pydantic
    try:
        from pydantic import BaseModel

        if isinstance(instance, BaseModel):
            # Pydantic instances have model_dump() method (Pydantic v2) or dict() method (Pydantic v1)
            if hasattr(instance, "model_dump"):
                return instance.model_dump()
            elif hasattr(instance, "dict"):
                return instance.dict()
            else:
                # Fallback to __dict__
                return {k: v for k, v in instance.__dict__.items() if not k.startswith("_")}
    except ImportError:
        pass

    raise TypeError(f"Expected SQLModel or Pydantic instance, got {type(instance)}")


def sqlmodel_to_dict(instance: Any) -> Dict[str, Any]:
    """Convert a SQLModel instance to a dictionary.

    Args:
        instance: SQLModel instance

    Returns:
        Dictionary representation of the instance

    Raises:
        TypeError: If instance is not a SQLModel instance
    """
    return model_to_dict(instance)


def dict_to_model(data: Dict[str, Any], model_class: Type) -> Any:
    """Convert a dictionary to a SQLModel or Pydantic instance.

    Args:
        data: Dictionary containing model data
        model_class: SQLModel or Pydantic model class

    Returns:
        SQLModel or Pydantic instance

    Raises:
        TypeError: If model_class is not a SQLModel or Pydantic class
        ImportError: If required dependencies are not installed
    """
    if is_sqlmodel_model(model_class):
        # SQLModel classes can be instantiated directly with dict unpacking
        return model_class(**data)
    elif is_pydantic_model(model_class):
        # Pydantic classes can be instantiated directly with dict unpacking
        return model_class(**data)
    else:
        raise TypeError(f"Expected SQLModel or Pydantic class, got {type(model_class)}")


def dict_to_sqlmodel(data: Dict[str, Any], model_class: Type) -> Any:
    """Convert a dictionary to a SQLModel instance.

    Args:
        data: Dictionary containing model data
        model_class: SQLModel model class

    Returns:
        SQLModel instance

    Raises:
        TypeError: If model_class is not a SQLModel class
        ImportError: If SQLModel is not installed
    """
    if not is_sqlmodel_model(model_class):
        raise TypeError(f"Expected SQLModel class, got {type(model_class)}")
    return dict_to_model(data, model_class)


def rows_to_models(rows: List[Dict[str, Any]], model_class: Type) -> List[Any]:
    """Convert a list of dictionaries to SQLModel or Pydantic instances.

    Args:
        rows: List of dictionaries representing rows
        model_class: SQLModel or Pydantic model class

    Returns:
        List of SQLModel or Pydantic instances

    Raises:
        TypeError: If model_class is not a SQLModel or Pydantic class
        ImportError: If required dependencies are not installed
    """
    if not is_model_class(model_class):
        raise TypeError(f"Expected SQLModel or Pydantic class, got {type(model_class)}")

    return [dict_to_model(row, model_class) for row in rows]


def rows_to_sqlmodels(rows: List[Dict[str, Any]], model_class: Type) -> List[Any]:
    """Convert a list of dictionaries to SQLModel instances.

    Args:
        rows: List of dictionaries representing rows
        model_class: SQLModel model class

    Returns:
        List of SQLModel instances

    Raises:
        TypeError: If model_class is not a SQLModel class
        ImportError: If SQLModel is not installed
    """
    if not is_sqlmodel_model(model_class):
        raise TypeError(f"Expected SQLModel class, got {type(model_class)}")
    return rows_to_models(rows, model_class)
