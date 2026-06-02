### Agents Configuration

```yaml
agents:
  - name: code_generator
    role: Develop and maintain the quantum state-vector simulator
    permissions:
      - read_files
      - write_files
      - execute_shell
      - network_access
  - name: test_runner
    role: Execute unit and integration tests
    permissions:
      - read_files
      - write_files
      - execute_shell
  - name: documentation_agent
    role: Maintain project documentation
    permissions:
      - read_files
      - write_files
```
