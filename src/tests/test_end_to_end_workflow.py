"""
End-to-end tests for the complete indexing and search workflow.

This test suite verifies the entire pipeline from source code to searchable
vector database, including:
- Full project indexing workflow
- Search functionality with intelligent chunking
- Cross-language project support
- Error handling throughout the pipeline
- Real-world usage scenarios
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import MCP tools for testing (using new modular structure)
from tools.indexing.index_tools import index_directory
from tools.indexing.search_tools import search


@pytest.mark.skip(reason="End-to-end tests require real Qdrant/Ollama services - need integration test setup")
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def sample_project(self):
        """Create a comprehensive sample project for testing."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir) / "sample_project"
        project_path.mkdir()

        # Create project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()

        # Main Python module
        main_py = project_path / "src" / "main.py"
        main_py.write_text(
            '''
"""Main application module."""

import os
import sys
from typing import List, Dict, Optional

class UserManager:
    """Manages user operations and authentication."""

    def __init__(self, db_path: str):
        """Initialize the user manager.

        Args:
            db_path: Path to the user database
        """
        self.db_path = db_path
        self.users: Dict[int, dict] = {}
        self._load_users()

    def _load_users(self) -> None:
        """Load users from the database file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.users = {int(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Error loading users: {e}")

    def create_user(self, name: str, email: str) -> int:
        """Create a new user.

        Args:
            name: User's name
            email: User's email address

        Returns:
            New user ID
        """
        user_id = max(self.users.keys(), default=0) + 1
        self.users[user_id] = {
            'id': user_id,
            'name': name,
            'email': email,
            'created_at': time.time()
        }
        self._save_users()
        return user_id

    def get_user(self, user_id: int) -> Optional[dict]:
        """Get user by ID.

        Args:
            user_id: ID of the user to retrieve

        Returns:
            User dictionary or None if not found
        """
        return self.users.get(user_id)

    def list_users(self) -> List[dict]:
        """Get all users.

        Returns:
            List of all users
        """
        return list(self.users.values())

    def _save_users(self) -> None:
        """Save users to the database file."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")

def main():
    """Main entry point."""
    user_manager = UserManager("/tmp/users.json")

    # Create some test users
    user1_id = user_manager.create_user("Alice", "alice@example.com")
    user2_id = user_manager.create_user("Bob", "bob@example.com")

    print(f"Created users: {user1_id}, {user2_id}")
    print(f"Total users: {len(user_manager.list_users())}")

if __name__ == "__main__":
    main()
'''
        )

        # Utility module
        utils_py = project_path / "src" / "utils.py"
        utils_py.write_text(
            '''
"""Utility functions for the application."""

import re
import hashlib
from typing import Any, Dict, List, Optional

def validate_email(email: str) -> bool:
    """Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def hash_password(password: str, salt: str = None) -> str:
    """Hash a password with salt.

    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)

    Returns:
        Hashed password
    """
    if salt is None:
        salt = os.urandom(32).hex()

    password_hash = hashlib.pbkdf2_hmac('sha256',
                                       password.encode('utf-8'),
                                       salt.encode('utf-8'),
                                       100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash.

    Args:
        password: Password to verify
        hashed: Hashed password to check against

    Returns:
        True if password matches
    """
    try:
        salt, hash_value = hashed.split(':')
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                           password.encode('utf-8'),
                                           salt.encode('utf-8'),
                                           100000)
        return password_hash.hex() == hash_value
    except ValueError:
        return False

def sanitize_input(text: str) -> str:
    """Sanitize user input.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>&"\'']', '', text)
    return sanitized.strip()

class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self.get_default_config()
            self.save_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config: {e}")
            self.config = self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "database_path": "/tmp/app.db",
            "max_users": 1000,
            "session_timeout": 3600,
            "debug": False
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self.save_config()

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
'''
        )

        # JavaScript frontend
        frontend_js = project_path / "src" / "frontend.js"
        frontend_js.write_text(
            """
/**
 * Frontend JavaScript for the user management application.
 */

class UserInterface {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.currentUser = null;
    }

    /**
     * Initialize the user interface
     */
    init() {
        this.setupEventListeners();
        this.loadUsers();
    }

    /**
     * Setup event listeners for UI elements
     */
    setupEventListeners() {
        const createUserBtn = document.getElementById('create-user');
        const userForm = document.getElementById('user-form');

        if (createUserBtn) {
            createUserBtn.addEventListener('click', () => this.showCreateUserForm());
        }

        if (userForm) {
            userForm.addEventListener('submit', (e) => this.handleCreateUser(e));
        }
    }

    /**
     * Load users from the API
     */
    async loadUsers() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/users`);
            const users = await response.json();
            this.displayUsers(users);
        } catch (error) {
            console.error('Error loading users:', error);
            this.showError('Failed to load users');
        }
    }

    /**
     * Display users in the interface
     * @param {Array} users - Array of user objects
     */
    displayUsers(users) {
        const userList = document.getElementById('user-list');
        if (!userList) return;

        userList.innerHTML = '';

        users.forEach(user => {
            const userElement = this.createUserElement(user);
            userList.appendChild(userElement);
        });
    }

    /**
     * Create a DOM element for a user
     * @param {Object} user - User object
     * @returns {HTMLElement} User element
     */
    createUserElement(user) {
        const div = document.createElement('div');
        div.className = 'user-item';
        div.innerHTML = `
            <h3>${user.name}</h3>
            <p>Email: ${user.email}</p>
            <p>ID: ${user.id}</p>
            <button onclick="userInterface.editUser(${user.id})">Edit</button>
            <button onclick="userInterface.deleteUser(${user.id})">Delete</button>
        `;
        return div;
    }

    /**
     * Show the create user form
     */
    showCreateUserForm() {
        const modal = document.getElementById('create-user-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }

    /**
     * Handle create user form submission
     * @param {Event} event - Form submit event
     */
    async handleCreateUser(event) {
        event.preventDefault();

        const formData = new FormData(event.target);
        const userData = {
            name: formData.get('name'),
            email: formData.get('email')
        };

        try {
            const response = await fetch(`${this.apiBaseUrl}/users`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userData)
            });

            if (response.ok) {
                this.hideCreateUserForm();
                this.loadUsers();
                this.showSuccess('User created successfully');
            } else {
                throw new Error('Failed to create user');
            }
        } catch (error) {
            console.error('Error creating user:', error);
            this.showError('Failed to create user');
        }
    }

    /**
     * Edit a user
     * @param {number} userId - User ID to edit
     */
    async editUser(userId) {
        // Implementation for editing users
        console.log(`Editing user ${userId}`);
    }

    /**
     * Delete a user
     * @param {number} userId - User ID to delete
     */
    async deleteUser(userId) {
        if (!confirm('Are you sure you want to delete this user?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/users/${userId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.loadUsers();
                this.showSuccess('User deleted successfully');
            } else {
                throw new Error('Failed to delete user');
            }
        } catch (error) {
            console.error('Error deleting user:', error);
            this.showError('Failed to delete user');
        }
    }

    /**
     * Show success message
     * @param {string} message - Success message
     */
    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        this.showMessage(message, 'error');
    }

    /**
     * Show a message to the user
     * @param {string} message - Message to show
     * @param {string} type - Message type (success, error, info)
     */
    showMessage(message, type) {
        const messageContainer = document.getElementById('messages');
        if (!messageContainer) return;

        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        messageElement.textContent = message;

        messageContainer.appendChild(messageElement);

        setTimeout(() => {
            messageElement.remove();
        }, 5000);
    }

    /**
     * Hide the create user form
     */
    hideCreateUserForm() {
        const modal = document.getElementById('create-user-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
}

// Initialize the user interface when the page loads
let userInterface;
document.addEventListener('DOMContentLoaded', () => {
    userInterface = new UserInterface('/api');
    userInterface.init();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UserInterface };
}
"""
        )

        # Configuration file
        config_json = project_path / "config.json"
        config_json.write_text(
            """
{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "userdb",
        "username": "admin",
        "password": "secret"
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "debug": false
    },
    "features": {
        "user_registration": true,
        "email_verification": true,
        "password_reset": true
    },
    "limits": {
        "max_users": 10000,
        "max_login_attempts": 5,
        "session_timeout": 3600
    }
}
"""
        )

        # Documentation
        readme_md = project_path / "README.md"
        readme_md.write_text(
            """
# User Management Application

A simple user management application with Python backend and JavaScript frontend.

## Features

- User creation and management
- Email validation
- Password hashing and verification
- Configuration management
- RESTful API

## Architecture

### Backend Components

- **UserManager**: Core user management logic
- **ConfigManager**: Configuration handling
- **Utils**: Utility functions for validation and security

### Frontend Components

- **UserInterface**: Main UI controller
- **Event Handlers**: User interaction management
- **API Client**: Backend communication

## Usage

### Creating Users

```python
from src.main import UserManager

manager = UserManager("/path/to/db.json")
user_id = manager.create_user("John Doe", "john@example.com")
```

### Validating Emails

```python
from src.utils import validate_email

is_valid = validate_email("user@example.com")
```

### Password Management

```python
from src.utils import hash_password, verify_password

hashed = hash_password("user_password")
is_correct = verify_password("user_password", hashed)
```

## Configuration

Edit `config.json` to customize application settings:

- Database connection parameters
- Server configuration
- Feature flags
- Limits and timeouts

## API Endpoints

- `GET /users` - List all users
- `POST /users` - Create new user
- `GET /users/{id}` - Get user by ID
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

## Testing

Run tests with:

```bash
pytest tests/
```

## Installation

1. Install dependencies
2. Configure database settings
3. Run the application

```bash
python src/main.py
```
"""
        )

        # Test file
        test_py = project_path / "tests" / "test_users.py"
        test_py.write_text(
            '''
"""Tests for user management functionality."""

import pytest
import tempfile
import os
from src.main import UserManager
from src.utils import validate_email, hash_password, verify_password

class TestUserManager:
    """Test user management operations."""

    def test_user_creation(self):
        """Test creating a new user."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            manager = UserManager(tmp.name)

            user_id = manager.create_user("Test User", "test@example.com")
            assert user_id > 0

            user = manager.get_user(user_id)
            assert user is not None
            assert user['name'] == "Test User"
            assert user['email'] == "test@example.com"

        os.unlink(tmp.name)

    def test_user_listing(self):
        """Test listing all users."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            manager = UserManager(tmp.name)

            # Create multiple users
            user1_id = manager.create_user("User 1", "user1@example.com")
            user2_id = manager.create_user("User 2", "user2@example.com")

            users = manager.list_users()
            assert len(users) == 2

            user_ids = [user['id'] for user in users]
            assert user1_id in user_ids
            assert user2_id in user_ids

        os.unlink(tmp.name)

class TestUtils:
    """Test utility functions."""

    def test_email_validation(self):
        """Test email validation."""
        assert validate_email("valid@example.com") == True
        assert validate_email("user.name@domain.co.uk") == True
        assert validate_email("invalid.email") == False
        assert validate_email("@example.com") == False
        assert validate_email("user@") == False

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"

        hashed = hash_password(password)
        assert hashed != password
        assert ':' in hashed  # Should contain salt and hash

        # Verify correct password
        assert verify_password(password, hashed) == True

        # Verify incorrect password
        assert verify_password("wrong_password", hashed) == False
'''
        )

        yield str(project_path)
        shutil.rmtree(temp_dir)

    def test_full_indexing_workflow(self, sample_project):
        """Test the complete indexing workflow from project to searchable database."""

        # Step 1: Index the project
        index_result = index_directory(directory=sample_project, clear_existing=True, incremental=False)

        # Verify indexing succeeded
        assert index_result["success"]
        assert index_result["total_files"] > 0
        assert index_result["total_points"] > 0
        assert len(index_result["collections"]) > 0

        print(f"Indexed {index_result['total_files']} files, generated {index_result['total_points']} chunks")

        # Step 2: Test search functionality
        search_queries = [
            "user management",
            "create user function",
            "email validation",
            "password hashing",
            "JavaScript UserInterface",
            "configuration management",
        ]

        for query in search_queries:
            search_result = search(query=query, n_results=5, cross_project=False, include_context=True)

            # Verify search returns results
            assert "results" in search_result
            assert len(search_result["results"]) > 0
            assert search_result["total"] > 0

            # Check result quality
            for result in search_result["results"]:
                assert "file_path" in result
                assert "content" in result
                assert "score" in result
                assert result["score"] > 0

                # Check for breadcrumb information
                if "breadcrumb" in result and result["breadcrumb"]:
                    assert isinstance(result["breadcrumb"], str)
                    assert len(result["breadcrumb"]) > 0

            print(f"Query '{query}': {len(search_result['results'])} results")

    def test_multi_language_search(self, sample_project):
        """Test search across multiple programming languages."""

        # Index the project
        index_result = index_directory(directory=sample_project, clear_existing=True)
        assert index_result["success"]

        # Test language-specific searches
        language_queries = [
            ("Python class definition", "python"),
            ("JavaScript function", "javascript"),
            ("JSON configuration", "json"),
        ]

        for query, expected_lang in language_queries:
            search_result = search(query=query, n_results=10)

            # Should find results
            assert len(search_result["results"]) > 0

            # Check if we found content from the expected language
            found_expected_lang = False
            for result in search_result["results"]:
                if result.get("language") == expected_lang:
                    found_expected_lang = True
                    break

            # At least some results should be from the expected language
            # (though cross-language results are also valuable)
            if not found_expected_lang:
                print(f"Warning: No {expected_lang} results for query '{query}'")

    def test_intelligent_chunking_quality(self, sample_project):
        """Test that intelligent chunking produces high-quality, semantic chunks."""

        # Index with intelligent chunking
        index_result = index_directory(directory=sample_project, clear_existing=True)
        assert index_result["success"]

        # Search for specific code constructs
        construct_queries = [
            "UserManager class",
            "create_user method",
            "validate_email function",
            "ConfigManager constructor",
            "UserInterface class",
        ]

        for query in construct_queries:
            search_result = search(query=query, n_results=3)

            assert len(search_result["results"]) > 0

            # Check that results contain relevant semantic information
            for result in search_result["results"]:
                content = result["content"].lower()

                # Should contain relevant keywords
                if "class" in query.lower():
                    assert "class" in content or "constructor" in content
                elif "method" in query.lower() or "function" in query.lower():
                    assert "def " in content or "function" in content or "=>" in content

                # Check for chunk metadata
                if "chunk_type" in result:
                    chunk_type = result["chunk_type"]
                    assert chunk_type in [
                        "function",
                        "class",
                        "method",
                        "interface",
                        "constant",
                    ]

                # Check for proper boundaries (chunks shouldn't be cut off mid-line)
                lines = result["content"].split("\n")
                if len(lines) > 1:
                    # First and last lines should be reasonably complete
                    assert len(lines[0].strip()) > 0
                    assert len(lines[-1].strip()) > 0

    def test_search_result_ranking(self, sample_project):
        """Test that search results are properly ranked by relevance."""

        # Index the project
        index_result = index_directory(directory=sample_project, clear_existing=True)
        assert index_result["success"]

        # Test specific queries where we can verify ranking
        test_cases = [
            {
                "query": "create user",
                "expected_high_relevance": ["create_user", "UserManager", "user"],
            },
            {
                "query": "email validation",
                "expected_high_relevance": ["validate_email", "email", "@"],
            },
            {
                "query": "password security",
                "expected_high_relevance": ["password", "hash", "pbkdf2"],
            },
        ]

        for test_case in test_cases:
            search_result = search(query=test_case["query"], n_results=10)

            assert len(search_result["results"]) > 0

            # Check that results are sorted by score (descending)
            scores = [result["score"] for result in search_result["results"]]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score"

            # Check that high-relevance terms appear in top results
            top_results = search_result["results"][:3]
            found_relevant_terms = []

            for result in top_results:
                content_lower = result["content"].lower()
                for term in test_case["expected_high_relevance"]:
                    if term.lower() in content_lower:
                        found_relevant_terms.append(term)

            # Should find at least some expected terms in top results
            assert len(found_relevant_terms) > 0, f"No relevant terms found for query '{test_case['query']}'"

    def test_incremental_indexing_workflow(self, sample_project):
        """Test incremental indexing preserves existing data and adds new content."""

        # Initial indexing
        initial_result = index_directory(directory=sample_project, clear_existing=True, incremental=False)
        assert initial_result["success"]
        initial_result["total_points"]

        # Add a new file to the project
        new_file = Path(sample_project) / "src" / "new_module.py"
        new_file.write_text(
            '''
"""New module added for incremental testing."""

class NewClass:
    """A new class for testing incremental indexing."""

    def new_method(self):
        """A new method."""
        return "new functionality"

def new_function():
    """A new function."""
    return "additional feature"
'''
        )

        # Incremental indexing
        incremental_result = index_directory(directory=sample_project, incremental=True)

        # Should detect and process the new file
        assert incremental_result["success"]

        # Should have more points than before (new content added)
        if "change_summary" in incremental_result:
            change_summary = incremental_result["change_summary"]
            if "files_added" in change_summary:
                assert change_summary["files_added"] > 0

        # Test that new content is searchable
        search_result = search(query="new functionality", n_results=5)
        assert len(search_result["results"]) > 0

        # Check that we can find the new class and method
        new_class_result = search(query="NewClass", n_results=3)
        assert len(new_class_result["results"]) > 0

        found_new_content = False
        for result in new_class_result["results"]:
            if "NewClass" in result["content"]:
                found_new_content = True
                break

        assert found_new_content, "Should find newly added content"

        # Clean up
        new_file.unlink()

    def test_error_handling_in_workflow(self, sample_project):
        """Test error handling throughout the indexing and search workflow."""

        # Add a file with syntax errors
        broken_file = Path(sample_project) / "src" / "broken.py"
        broken_file.write_text(
            '''
# File with syntax errors for testing

def broken_function(
    # Missing closing parenthesis
    return "broken"

class IncompleteClass
    # Missing colon
    def method(self):
        return True

def valid_function():
    """This function should still be indexed."""
    return "valid content"
'''
        )

        try:
            # Index project with broken file
            index_result = index_directory(directory=sample_project, clear_existing=True)

            # Should still succeed overall despite syntax errors
            assert index_result["success"]
            assert index_result["total_points"] > 0

            # Should be able to find the valid content
            search_result = search(query="valid content", n_results=5)
            assert len(search_result["results"]) > 0

            # Check if error information is available
            if "errors" in index_result:
                errors = index_result["errors"]
                # Should have some error count but not be completely failed
                assert errors["count"] >= 0

        finally:
            # Clean up
            if broken_file.exists():
                broken_file.unlink()

    def test_cross_project_search(self, sample_project):
        """Test search across multiple projects."""

        # Create a second project
        temp_dir = Path(sample_project).parent
        second_project = temp_dir / "second_project"
        second_project.mkdir()

        # Add a file to the second project
        second_file = second_project / "different.py"
        second_file.write_text(
            '''
"""Different project with unique content."""

class SecondProjectClass:
    """Class from the second project."""

    def unique_method(self):
        """A method with unique terminology."""
        return "distinctive_functionality"

UNIQUE_CONSTANT = "second_project_value"
'''
        )

        try:
            # Index both projects
            index_directory(directory=sample_project, clear_existing=True)
            index_directory(directory=str(second_project), clear_existing=True)

            # Test project-specific search (should only find first project)
            project_search = search(query="UserManager", cross_project=False, n_results=5)

            # Should find results from the current project
            assert len(project_search["results"]) > 0
            assert project_search["search_scope"] == "current project"

            # Test cross-project search
            cross_search = search(query="distinctive_functionality", cross_project=True, n_results=5)

            # Should be able to search across projects
            # (Results depend on which project context we're in)
            assert "search_scope" in cross_search

        finally:
            # Clean up
            shutil.rmtree(second_project)

    def test_large_project_performance(self, sample_project):
        """Test performance with a larger project structure."""

        # Expand the project with more files
        src_dir = Path(sample_project) / "src"

        # Create multiple modules
        for i in range(20):
            module_file = src_dir / f"module_{i}.py"
            module_file.write_text(
                f'''
"""Module {i} for performance testing."""

class Module{i}Class:
    """Class in module {i}."""

    def __init__(self):
        self.module_id = {i}

    def process_{i}(self, data):
        """Process data in module {i}."""
        return f"Module {i} processed: {{data}}"

    def calculate_{i}(self, x, y):
        """Calculate result in module {i}."""
        return x + y + {i}

def module_{i}_function():
    """Function in module {i}."""
    return f"Module {i} result"

MODULE_{i}_CONSTANT = {i} * 100
'''
            )

        try:
            # Time the indexing operation
            start_time = time.time()
            index_result = index_directory(directory=sample_project, clear_existing=True)
            indexing_time = time.time() - start_time

            # Should complete within reasonable time
            assert indexing_time < 30, f"Indexing took too long: {indexing_time:.2f}s"
            assert index_result["success"]
            assert index_result["total_points"] > 50  # Should have many chunks

            # Test search performance
            start_time = time.time()
            search_result = search(query="process data", n_results=10)
            search_time = time.time() - start_time

            # Search should be fast
            assert search_time < 5, f"Search took too long: {search_time:.2f}s"
            assert len(search_result["results"]) > 0

            print(f"Performance: Indexing {indexing_time:.2f}s, Search {search_time:.2f}s")

        finally:
            # Clean up generated files
            for i in range(20):
                module_file = src_dir / f"module_{i}.py"
                if module_file.exists():
                    module_file.unlink()


@pytest.mark.skip(reason="End-to-end tests require real Qdrant/Ollama services - need integration test setup")
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_documentation_search(self, sample_project):
        """Test searching through documentation and code comments."""

        # Index the project
        index_result = index_directory(directory=sample_project, clear_existing=True)
        assert index_result["success"]

        # Search for documentation content
        doc_queries = [
            "user management application",
            "API endpoints",
            "configuration settings",
            "installation instructions",
        ]

        for query in doc_queries:
            search_result = search(query=query, n_results=5)

            # Should find documentation content
            assert len(search_result["results"]) > 0

            # Check that markdown content is properly indexed
            found_markdown = False
            for result in search_result["results"]:
                if ".md" in result.get("file_path", ""):
                    found_markdown = True
                    break

            if not found_markdown:
                print(f"No markdown results for query: {query}")

    def test_configuration_search(self, sample_project):
        """Test searching through configuration files."""

        # Index the project
        index_result = index_directory(directory=sample_project, clear_existing=True)
        assert index_result["success"]

        # Search for configuration content
        config_queries = [
            "database configuration",
            "server port",
            "session timeout",
            "feature flags",
        ]

        for query in config_queries:
            search_result = search(query=query, n_results=5)

            # Should find configuration content
            assert len(search_result["results"]) > 0

            # Check for JSON content
            found_config = False
            for result in search_result["results"]:
                if "config" in result.get("file_path", "").lower() or ".json" in result.get("file_path", ""):
                    found_config = True
                    break

            if not found_config:
                print(f"No config results for query: {query}")

    def test_code_discovery_workflow(self, sample_project):
        """Test workflow for discovering code patterns and structures."""

        # Index the project
        index_result = index_directory(directory=sample_project, clear_existing=True)
        assert index_result["success"]

        # Test discovery queries that developers might use
        discovery_queries = [
            "how to create a user",
            "password validation logic",
            "error handling patterns",
            "database connection setup",
            "API endpoint implementation",
        ]

        for query in discovery_queries:
            search_result = search(query=query, n_results=10, include_context=True)

            # Should find relevant code
            assert len(search_result["results"]) > 0

            # Results should include code context
            for result in search_result["results"]:
                assert len(result["content"]) > 50  # Should have substantial content

                # Check for code-like content
                content = result["content"]
                has_code_markers = any(
                    marker in content
                    for marker in [
                        "def ",
                        "class ",
                        "function",
                        "import",
                        "return",
                        "{",
                        "}",
                    ]
                )

                if not has_code_markers:
                    print(f"Result may not contain code: {result['file_path']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
