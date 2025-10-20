import os
import json
import yaml
import tempfile
import pytest
from unittest.mock import patch, mock_open
from jinja2 import Template

from src.ai_factory_model.llm.model_utils import (
    load_from_file,
    create_template,
    read_template,
    render_template,
    SEP_PATTERN
)


class TestLoadFromFile:
    """Test cases for load_from_file function"""

    def test_load_json_file_success(self):
        """Test successfully loading a JSON file"""
        test_data = {"key": "value", "number": 123}
        json_content = json.dumps(test_data)

        with patch("builtins.open", mock_open(read_data=json_content)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".json")):
                    result = load_from_file("test.json")
                    assert result == test_data

    def test_load_yaml_file_success(self):
        """Test successfully loading a YAML file"""
        test_data = {"key": "value", "number": 123}
        yaml_content = yaml.safe_dump(test_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".yaml")):
                    result = load_from_file("test.yaml")
                    assert result == test_data

    def test_load_yml_file_success(self):
        """Test successfully loading a .yml file"""
        test_data = {"key": "value", "number": 123}
        yaml_content = yaml.safe_dump(test_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".yml")):
                    result = load_from_file("test.yml")
                    assert result == test_data

    def test_load_prompt_file_success(self):
        """Test successfully loading a .prompt file"""
        prompt_content = "This is a test prompt"

        with patch("builtins.open", mock_open(read_data=prompt_content)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".prompt")):
                    result = load_from_file("test.prompt")
                    assert result == prompt_content

    def test_load_file_not_exists(self):
        """Test loading a file that doesn't exist"""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_from_file("nonexistent.json")
            assert "does not exist" in str(exc_info.value)

    def test_load_unsupported_file_format(self):
        """Test loading a file with unsupported extension"""
        with patch("os.path.exists", return_value=True):
            with patch("os.path.splitext", return_value=("test", ".txt")):
                with patch("builtins.open", mock_open(read_data="content")):
                    with pytest.raises(RuntimeError) as exc_info:
                        load_from_file("test.txt")
                    assert "Error loading file" in str(exc_info.value)
                    assert "Not supported file format" in str(exc_info.value)

    def test_load_invalid_json(self):
        """Test loading invalid JSON content"""
        invalid_json = "{ invalid json content"

        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".json")):
                    with pytest.raises(RuntimeError) as exc_info:
                        load_from_file("test.json")
                    assert "Error loading file" in str(exc_info.value)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML content"""
        invalid_yaml = "key: value\n  invalid: yaml: content:"

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".yaml")):
                    with pytest.raises(RuntimeError) as exc_info:
                        load_from_file("test.yaml")
                    assert "Error loading file" in str(exc_info.value)

    def test_load_file_permission_error(self):
        """Test loading a file with permission error"""
        with patch("os.path.exists", return_value=True):
            with patch("os.path.splitext", return_value=("test", ".json")):
                with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                    with pytest.raises(RuntimeError) as exc_info:
                        load_from_file("test.json")
                    assert "Error loading file" in str(exc_info.value)
                    assert "Permission denied" in str(exc_info.value)

    @patch("src.ai_factory_model.llm.model_utils.debug")
    def test_load_file_logs_debug(self, mock_info):
        """Test that loading a file logs the appropriate debug message"""
        test_data = {"key": "value"}
        json_content = json.dumps(test_data)

        with patch("builtins.open", mock_open(read_data=json_content)):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.splitext", return_value=("test", ".json")):
                    load_from_file("test.json")
                    mock_info.assert_called_once_with('Loading from file "test.json"')

    def test_load_direct_value_error(self):
        """Test that ValueError from unsupported format is properly wrapped in RuntimeError"""
        with patch("os.path.exists", return_value=True):
            with patch("os.path.splitext", return_value=("test", ".unsupported")):
                with patch("builtins.open", mock_open(read_data="content")):
                    with pytest.raises(RuntimeError) as exc_info:
                        load_from_file("test.unsupported")
                    # Verify that the original ValueError is wrapped
                    assert "Error loading file" in str(exc_info.value)
                    assert "Not supported file format" in str(exc_info.value)


class TestCreateTemplate:
    """Test cases for create_template function"""

    def test_create_template_success(self):
        """Test successfully creating a template from a file"""
        template_content = f"Hello {{{{ name }}}}{SEP_PATTERN}Your message: {{{{ message }}}}"

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            assert isinstance(template, Template)
            # Test that the template can be rendered
            result = template.render(name="World", message="test")
            assert "Hello World" in result
            assert "Your message: test" in result

    def test_create_template_with_whitespace_control(self):
        """Test that create_template uses trim_blocks and lstrip_blocks"""
        template_content = """
        {%- if condition %}
            System message
        {%- endif %}
        --- message ---
        User input"""

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            result = template.render(condition=True)
            # Verify whitespace control is working
            assert "System message" in result
            assert result.count("\n") < template_content.count("\n")  # Should have fewer newlines

    def test_create_template_empty_content(self):
        """Test creating template with empty content"""
        template_content = ""

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            assert isinstance(template, Template)
            result = template.render()
            assert result == ""

    def test_create_template_with_variables(self):
        """Test creating template with various Jinja2 variables"""
        template_content = (
            f"Count: {{{{ count }}}} Items: {{% for item in items %}}{{{{ item }}}} {{% endfor %}}"
            f"{SEP_PATTERN}Total: {{{{ count }}}}"
        )

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            result = template.render(count=3, items=["a", "b", "c"])
            assert "Count: 3" in result
            assert "Items: a b c" in result
            assert "Total: 3" in result

    def test_create_template_with_complex_jinja(self):
        """Test creating template with complex Jinja2 syntax"""
        sep_pattern = "--- message ---"
        template_content = f"""
        {{% set greeting = "Hello" %}}
        {{{{ greeting }}}} {{{{ name|default("User") }}}}!
        {{% if messages %}}
        Messages:
        {{% for msg in messages %}}
        - {{{{ msg }}}}
        {{% endfor %}}
        {{% endif %}}
        {sep_pattern}
        Response needed"""

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            result = template.render(name="John", messages=["Hi", "How are you?"])
            assert "Hello John!" in result
            assert "Messages:" in result
            assert "- Hi" in result
            assert "- How are you?" in result

    def test_create_template_file_not_found(self):
        """Test creating template when file doesn't exist"""
        with patch("src.ai_factory_model.llm.model_utils.load_from_file",
                   side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                create_template("nonexistent.prompt")

    def test_create_template_invalid_file_format(self):
        """Test creating template with invalid file format"""
        with patch("src.ai_factory_model.llm.model_utils.load_from_file",
                   side_effect=RuntimeError("Error loading file: Not supported file format")):
            with pytest.raises(RuntimeError) as exc_info:
                create_template("test.invalid")
            assert "Error loading file" in str(exc_info.value)

    def test_create_template_file_permission_error(self):
        """Test creating template with file permission error"""
        with patch("src.ai_factory_model.llm.model_utils.load_from_file",
                   side_effect=RuntimeError("Error loading file: Permission denied")):
            with pytest.raises(RuntimeError) as exc_info:
                create_template("test.prompt")
            assert "Permission denied" in str(exc_info.value)

    def test_create_template_with_invalid_jinja_syntax(self):
        """Test creating template with invalid Jinja2 syntax"""
        template_content = "Hello {{{{ invalid syntax"

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            # Template creation should fail with invalid syntax
            from jinja2.exceptions import TemplateSyntaxError
            with pytest.raises(TemplateSyntaxError):
                create_template("test.prompt")

    def test_create_template_return_type(self):
        """Test that create_template returns correct Template type"""
        template_content = f"Simple template{SEP_PATTERN}content"

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            assert isinstance(template, Template)
            assert hasattr(template, 'render')
            assert hasattr(template, 'generate')
            assert callable(template.render)

    def test_create_template_integration_with_render_template(self):
        """Test create_template integration with render_template function"""
        template_content = f"System: {{{{ system_msg }}}}{SEP_PATTERN}User: {{{{ user_msg }}}}"
        params = {"system_msg": "Hello", "user_msg": "World"}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            template = create_template("test.prompt")
            system, user_input = render_template(template, params)
            assert system == "System: Hello"
            assert user_input == "User: World"


class TestReadTemplate:
    """Test cases for read_template function"""

    def test_read_template_success(self):
        """Test successfully reading and processing a template"""
        template_content = f"System message{SEP_PATTERN}User input"
        params = {"variable": "value"}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            system, user_input = read_template("test.prompt", params)
            assert system == "System message"
            assert user_input == "User input"

    def test_read_template_with_jinja_variables(self):
        """Test reading a template with Jinja2 variables"""
        template_content = f"Hello {{{{ name }}}} system{SEP_PATTERN}Your message: {{{{ message }}}}"
        params = {"name": "World", "message": "test"}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            system, user_input = read_template("test.prompt", params)
            assert system == "Hello World system"
            assert user_input == "Your message: test"

    def test_read_template_missing_separator(self):
        """Test reading a template without separator pattern"""
        template_content = "Only system message without separator"
        params = {}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            with pytest.raises(ValueError) as exc_info:
                read_template("test.prompt", params)
            assert "Template separator not found" in str(exc_info.value)

    def test_read_template_empty_params(self):
        """Test reading a template with empty parameters"""
        template_content = f"System message{SEP_PATTERN}User input"
        params = {}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            system, user_input = read_template("test.prompt", params)
            assert system == "System message"
            assert user_input == "User input"

    def test_read_template_with_whitespace_control(self):
        """Test that template uses trim_blocks and lstrip_blocks"""
        template_content = f"  System message  \n  {SEP_PATTERN}  \n  User input  "
        params = {}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            system, user_input = read_template("test.prompt", params)
            # The exact behavior depends on Jinja2's whitespace control
            assert SEP_PATTERN not in system
            assert SEP_PATTERN not in user_input

    def test_read_template_separator_validation(self):
        """Test that read_template properly validates separator presence"""
        template_content = "Template with no separator pattern anywhere"
        params = {"var": "value"}

        with patch("src.ai_factory_model.llm.model_utils.load_from_file", return_value=template_content):
            with pytest.raises(ValueError) as exc_info:
                read_template("test.prompt", params)
            assert str(exc_info.value) == "Template separator not found: \"--- message ---\""
            assert exc_info.type == ValueError


class TestRenderTemplate:
    """Test cases for render_template function"""

    def test_render_template_success(self):
        """Test successfully rendering a template"""
        template_content = f"Hello {{{{ name }}}}{SEP_PATTERN}Message: {{{{ message }}}}"
        template = Template(template_content)
        params = {"name": "World", "message": "test"}

        system, user_input = render_template(template, params)
        assert system == "Hello World"
        assert user_input == "Message: test"

    def test_render_template_no_variables(self):
        """Test rendering a template without variables"""
        template_content = f"Static system message{SEP_PATTERN}Static user input"
        template = Template(template_content)
        params = {}

        system, user_input = render_template(template, params)
        assert system == "Static system message"
        assert user_input == "Static user input"

    def test_render_template_missing_separator(self):
        """Test rendering a template without separator pattern"""
        template_content = "Only system message without separator"
        template = Template(template_content)
        params = {}

        with pytest.raises(ValueError) as exc_info:
            render_template(template, params)
        assert "Template separator not found" in str(exc_info.value)

    def test_render_template_multiple_separators(self):
        """Test rendering a template with multiple separator patterns"""
        template_content = f"System{SEP_PATTERN}Middle{SEP_PATTERN}End"
        template = Template(template_content)
        params = {}

        system, user_input = render_template(template, params)
        assert system == "System"  # Finds first separator
        assert user_input == f"Middle{SEP_PATTERN}End"  # Everything after first separator

    def test_render_template_with_complex_jinja(self):
        """Test rendering a template with complex Jinja2 syntax"""
        template_content = (
            f"System: {{% for item in items %}}{{{{ item }}}} {{% endfor %}}"
            f"{SEP_PATTERN}User: {{{{ user_name }}}}"
        )
        template = Template(template_content)
        params = {"items": ["a", "b", "c"], "user_name": "John"}

        system, user_input = render_template(template, params)
        assert "a b c" in system
        assert user_input == "User: John"

    def test_render_template_missing_variable(self):
        """Test rendering a template with missing variable"""
        template_content = f"Hello {{{{ missing_var }}}}{SEP_PATTERN}Input"
        template = Template(template_content)
        params = {}

        # Jinja2 by default renders undefined variables as empty strings
        system, user_input = render_template(template, params)
        assert system == "Hello "  # "Hello " with undefined variable rendered as empty
        assert user_input == "Input"

    def test_render_template_separator_validation(self):
        """Test that render_template properly validates separator presence"""
        template_content = "No separator here at all"
        template = Template(template_content)
        params = {}

        with pytest.raises(ValueError) as exc_info:
            render_template(template, params)
        assert str(exc_info.value) == "Template separator not found: \"--- message ---\""
        assert exc_info.type == ValueError


class TestIntegration:
    """Integration tests for model_utils functions"""

    def test_full_workflow_with_real_files(self):
        """Test the complete workflow with actual temporary files"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_data = {"test": "data"}
            json.dump(json_data, f)
            json_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {"yaml": "content"}
            yaml.safe_dump(yaml_data, f)
            yaml_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.prompt', delete=False) as f:
            template_content = f"System: {{{{ system_msg }}}}{SEP_PATTERN}User: {{{{ user_msg }}}}"
            f.write(template_content)
            prompt_file = f.name

        try:
            # Test JSON loading
            result = load_from_file(json_file)
            assert result == json_data

            # Test YAML loading
            result = load_from_file(yaml_file)
            assert result == yaml_data

            # Test template reading
            params = {"system_msg": "Hello", "user_msg": "World"}
            system, user_input = read_template(prompt_file, params)
            assert "Hello" in system
            assert "World" in user_input

        finally:
            # Cleanup
            os.unlink(json_file)
            os.unlink(yaml_file)
            os.unlink(prompt_file)

    def test_sep_pattern_constant(self):
        """Test that SEP_PATTERN constant is correctly defined"""
        assert SEP_PATTERN == "--- message ---"
        assert isinstance(SEP_PATTERN, str)
        assert len(SEP_PATTERN) > 0
