# Best Practices for Codebase RAG MCP

A comprehensive guide for effective use of the Codebase RAG MCP Server, including search optimization, project organization, and cross-project workflows.

## Effective Search Queries

### Language-Specific Search Patterns

#### Python
```
# Function-focused queries
"Find Python functions that handle database connections"
"Show me error handling patterns in Python using try-except"
"Locate async functions that process files"

# Class and method queries
"Find Python classes that inherit from BaseModel"
"Show me methods that use decorators like @property"
"Locate __init__ methods with dependency injection"

# Framework-specific
"Find Flask route handlers that return JSON"
"Show me Django models with foreign key relationships"
"Locate FastAPI endpoints with authentication"
```

#### JavaScript/TypeScript
```
# React component queries
"Find React components that use useState hook"
"Show me TypeScript interfaces for API responses"
"Locate components that handle form submission"

# Framework patterns
"Find Express middleware functions"
"Show me Angular services with dependency injection"
"Locate Vue components with computed properties"

# Modern JS patterns
"Find functions using async/await patterns"
"Show me arrow functions that return promises"
"Locate destructuring assignments in function parameters"
```

#### Go
```
# Go-specific patterns
"Find Go functions that return errors"
"Show me struct methods with pointer receivers"
"Locate goroutines that use channels"

# Package and interface queries
"Find interfaces with HTTP handler methods"
"Show me functions that implement io.Reader interface"
"Locate context.Context usage in functions"
```

#### Java
```
# Object-oriented patterns
"Find Java classes that implement Serializable"
"Show me methods with @Override annotation"
"Locate constructors with dependency injection"

# Framework-specific
"Find Spring Boot controller methods"
"Show me JPA entity classes with relationships"
"Locate methods with @Transactional annotation"
```

### Query Construction Best Practices

#### Use Specific Technical Terms
```
✅ Good: "Find functions that validate JWT tokens"
❌ Vague: "Find authentication code"

✅ Good: "Show me React components that use useEffect with cleanup"
❌ Vague: "Find React hooks"
```

#### Combine Functional and Technical Aspects
```
✅ Good: "Find database migration scripts that create indexes"
✅ Good: "Show me API endpoints that handle file uploads"
✅ Good: "Locate error handling functions that log to external services"
```

#### Use Context-Aware Language
```
✅ Good: "Find functions that parse CSV files and return structured data"
✅ Good: "Show me classes that implement the Observer pattern"
✅ Good: "Locate middleware that handles CORS configuration"
```

## Project Organization Strategies

### Single Project Workflow

#### Initial Setup
1. **Analyze Before Indexing**
   ```
   # Get repository overview
   analyze_repository_tool(directory="/path/to/project")

   # Check filtering configuration
   get_file_filtering_stats_tool(directory="/path/to/project")
   ```

2. **Configure Filtering**
   - Create `.ragignore` file to exclude:
     - Build directories (`build/`, `dist/`, `target/`)
     - Dependency directories (`node_modules/`, `venv/`, `.venv/`)
     - Generated files (`*.pyc`, `*.class`, `*.o`)
     - Large data files (`*.csv`, `*.json` > 1MB)

3. **Initial Indexing**
   ```
   # Full indexing with clear start
   index_directory(
     directory="/path/to/project",
     clear_existing=true,
     project_name="my-project"
   )
   ```

#### Ongoing Maintenance
1. **Incremental Updates**
   ```
   # Regular incremental updates
   index_directory(
     directory="/path/to/project",
     incremental=true
   )
   ```

2. **Monitor Performance**
   ```
   # Check indexing metrics
   get_chunking_metrics_tool()

   # Verify parser health
   diagnose_parser_health_tool()
   ```

### Multi-Project Workflow

#### Project Naming Strategy
- Use descriptive, hierarchical names:
  - `company-api-core`
  - `frontend-react-dashboard`
  - `mobile-ios-app`
  - `shared-utils-library`

#### Cross-Project Search Strategies

1. **Global Knowledge Base Queries**
   ```
   # Search across all projects
   search(
     query="Find authentication implementation patterns",
     cross_project=true,
     n_results=10
   )
   ```

2. **Targeted Project Searches**
   ```
   # Search specific projects for focused analysis
   search(
     query="Find authentication implementation patterns",
     target_projects=["backend-api", "user-service"],
     n_results=10
   )

   # Compare patterns between frontend and backend
   search(
     query="Show me error handling patterns",
     target_projects=["frontend-react", "backend-node"],
     include_context=true
   )

   # Agent knowledge acquisition from specific project
   search(
     query="Show me all API endpoint patterns and middleware",
     target_projects=["main-api-service"],
     n_results=20,
     context_chunks=2
   )
   ```

3. **Comparative Analysis**
   ```
   # Compare implementations across projects
   search(
     query="Show me different database connection patterns",
     cross_project=true,
     search_mode="semantic"
   )

   # Compare specific projects only
   search(
     query="Compare authentication middleware implementations",
     target_projects=["service-a", "service-b", "legacy-service"],
     search_mode="hybrid"
   )
   ```

4. **Architecture Pattern Discovery**
   ```
   # Find common architectural patterns across all projects
   search(
     query="Locate dependency injection containers",
     cross_project=true,
     include_context=true
   )

   # Focus on microservices architecture patterns
   search(
     query="Find service communication patterns",
     target_projects=["user-service", "payment-service", "notification-service"],
     include_context=true
   )
   ```

5. **Agent Knowledge Transfer**
   ```
   # Give Agent comprehensive knowledge of a specific project
   search(
     query="Show me the complete application architecture and key patterns",
     target_projects=["core-application"],
     n_results=50,
     include_context=true,
     context_chunks=3
   )

   # Load Agent with knowledge from related projects for development context
   search(
     query="Find all configuration patterns and deployment scripts",
     target_projects=["infrastructure", "deployment-configs"],
     n_results=30
   )
   ```

#### Project Relationship Management

1. **Related Projects**
   - Index related projects together
   - Use consistent naming conventions
   - Maintain cross-references in documentation

2. **Shared Libraries**
   - Index shared/common libraries separately
   - Tag with descriptive project names
   - Update frequently as they change often

## Performance Optimization

### Indexing Performance

#### Resource Management
```
# Configure for your system
INDEXING_CONCURRENCY=4          # CPU cores - 1
INDEXING_BATCH_SIZE=20          # Files per batch
EMBEDDING_BATCH_SIZE=10         # Embeddings per API call
MEMORY_WARNING_THRESHOLD_MB=1000 # Adjust for available RAM
```

#### Large Codebase Strategies
1. **Staged Indexing**
   - Index core modules first
   - Add peripheral code in phases
   - Use incremental updates for maintenance

2. **Selective Indexing**
   - Focus on actively developed areas
   - Exclude archived or legacy code
   - Use pattern matching for specific file types

3. **Resource Monitoring**
   ```
   # Monitor progress during large operations
   get_indexing_progress_tool()

   # Check system health
   health_check()
   ```

### Search Performance

#### Query Optimization
1. **Search Mode Selection**
   - `semantic`: Best for conceptual queries
   - `keyword`: Best for exact term matches
   - `hybrid`: Balanced approach (recommended default)

2. **Result Scoping**
   - Use appropriate `n_results` values (5-20 typical)
   - Enable cross-project search only when needed
   - Adjust context chunks based on use case

3. **Context Management**
   ```
   # Minimal context for quick overviews
   search(query="...", include_context=false)

   # Rich context for understanding relationships
   search(query="...", context_chunks=2)
   ```

#### Reranking Configuration (Two-Stage RAG)

The search tool uses two-stage RAG with cross-encoder reranking by default, improving accuracy by 22-31%.

1. **When to Use Reranking (Default)**
   - Complex semantic queries requiring precise understanding
   - Architectural analysis and code understanding tasks
   - When accuracy is more important than speed
   ```
   # High-precision search (reranking enabled by default)
   search(
     query="Find authentication implementation patterns",
     n_results=10
   )
   ```

2. **When to Disable Reranking**
   - Speed-critical applications requiring fast responses
   - Simple keyword-based searches
   - Large result sets (>50 results)
   ```
   # Fast search without reranking
   search(
     query="Find all API endpoints",
     enable_reranking=false,
     n_results=30
   )
   ```

3. **Tuning Rerank Parameters**
   - Higher `rerank_top_k` (50-100) improves quality but increases latency
   - Lower `rerank_top_k` (20-30) for faster results with moderate quality
   ```
   # More candidates for complex queries
   search(
     query="Complex architectural pattern analysis",
     rerank_top_k=100,
     n_results=15
   )
   ```

4. **Performance Characteristics**
   - Reranking latency: ~100ms (Apple Silicon/MPS) to ~400ms (CPU)
   - Quality improvement: 22-31% better accuracy
   - First search may be slower (model loading)

## Knowledge Base Usage Patterns

### Development Workflows

#### Code Discovery
1. **Finding Entry Points**
   ```
   search(query="Find main application entry points")
   search(query="Locate CLI command definitions")
   search(query="Show me HTTP route handlers")
   ```

2. **Understanding Data Flow**
   ```
   search(query="Find functions that process user input")
   search(query="Show me database query builders")
   search(query="Locate data validation functions")
   ```

3. **Architectural Understanding**
   ```
   search(query="Find dependency injection configurations")
   search(query="Show me factory pattern implementations")
   search(query="Locate service layer interfaces")
   ```

#### Debugging and Maintenance
1. **Error Investigation**
   ```
   search(query="Find error handling patterns for network failures")
   search(query="Show me logging statements with error levels")
   search(query="Locate exception handling in async functions")
   ```

2. **Performance Analysis**
   ```
   search(query="Find functions with caching implementations")
   search(query="Show me database query optimization patterns")
   search(query="Locate memory management code")
   ```

#### Feature Development
1. **Pattern Replication**
   ```
   search(query="Find similar feature implementations")
   search(query="Show me test patterns for API endpoints")
   search(query="Locate configuration management examples")
   ```

2. **Integration Examples**
   ```
   search(query="Find third-party service integration patterns")
   search(query="Show me API client implementations")
   search(query="Locate authentication middleware examples")
   ```

### Team Collaboration

#### Knowledge Sharing
1. **Onboarding New Team Members**
   - Create query collections for common patterns
   - Document project-specific search strategies
   - Share effective query examples

2. **Code Review Assistance**
   ```
   search(query="Find similar error handling patterns")
   search(query="Show me established coding patterns for this feature")
   ```

3. **Architecture Decisions**
   ```
   search(query="Find existing design pattern implementations")
   search(query="Show me performance optimization examples")
   ```

## Troubleshooting Common Issues

### Debugging with File Logging

Enable file logging to diagnose issues:

```bash
# In .env file
LOG_FILE_ENABLED=true
LOG_LEVEL=DEBUG
LOG_FILE_PATH=logs/debug.log
```

#### Useful Log Analysis Commands
```bash
# Watch logs in real-time during indexing
tail -f logs/debug.log

# Find errors
grep "ERROR" logs/debug.log

# Check reranking performance
grep "reranking" logs/debug.log

# Find slow operations
grep "duration" logs/debug.log | grep -E "[0-9]{2,}\.[0-9]+s"
```

Logs are automatically rotated at 10MB with 5 backup files. See [CONFIGURATION.md](./CONFIGURATION.md#logging) for options.

### Indexing Problems

#### Slow Indexing Performance
1. **Check System Resources**
   ```
   health_check_tool()
   get_indexing_progress_tool()
   ```

2. **Optimize Configuration**
   - Reduce batch sizes for memory-constrained systems
   - Increase concurrency for CPU-bound operations
   - Exclude unnecessary files with `.ragignore`

3. **Monitor Progress**
   - Enable file logging for detailed insights (`LOG_FILE_ENABLED=true`)
   - Check chunking metrics for bottlenecks
   - Verify parser health for syntax errors

#### Incomplete or Failed Indexing
1. **Syntax Error Handling**
   ```
   diagnose_parser_health_tool(comprehensive=true)
   get_chunking_metrics_tool()
   ```

2. **File Access Issues**
   - Check file permissions
   - Verify directory accessibility
   - Review `.ragignore` patterns

3. **Memory or Disk Space**
   - Monitor system resources
   - Clear temporary files
   - Use incremental indexing for large projects

### Search Quality Issues

#### Poor Search Results
1. **Query Refinement**
   - Use more specific technical terms
   - Include context about expected results
   - Try different search modes

2. **Index Quality**
   ```
   check_index_status()
   get_file_metadata_tool(file_path="problematic/file.py")
   ```

3. **Language Support**
   - Verify language is supported
   - Check file extension recognition
   - Review parser diagnostics

#### Missing Expected Results
1. **Verify Indexing**
   ```
   search(query="filename:specific_file.py")
   get_project_info_tool()
   ```

2. **Check Filtering**
   ```
   get_file_filtering_stats_tool()
   analyze_repository_tool()
   ```

3. **Reindex if Necessary**
   ```
   reindex_file_tool(file_path="path/to/file.py")
   ```

### System Integration Issues

#### MCP Server Connectivity
1. **Health Verification**
   ```
   health_check()
   ```

2. **Service Dependencies**
   - Verify Qdrant is running and accessible
   - Check Ollama service and models
   - Test network connectivity

3. **Configuration Issues**
   - Review environment variables
   - Check MCP server registration
   - Verify CLI tool integration

## Advanced Techniques

### Custom Search Strategies

#### Multi-Step Discovery
1. Start with broad conceptual searches
2. Narrow down to specific implementations
3. Explore related code with context

#### Comparative Analysis
1. Search for similar patterns across projects
2. Analyze implementation differences
3. Extract best practices and standards

#### Architectural Exploration
1. Map system boundaries and interfaces
2. Trace data flow through components
3. Identify integration points and dependencies

### Automation and Integration

#### Search-Driven Development
- Use searches to find similar implementations before coding
- Validate architectural decisions with existing patterns
- Discover reusable components and utilities

#### Documentation Generation
- Extract function signatures and documentation
- Generate API documentation from search results
- Create architectural diagrams from code relationships

#### Quality Assurance
- Find inconsistent implementation patterns
- Locate outdated or deprecated code
- Identify security or performance antipatterns
