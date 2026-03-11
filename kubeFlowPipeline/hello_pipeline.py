from kfp import dsl
from kfp import compiler
from kfp import kubernetes

# Define a simple component using a Python function
@dsl.component
def say_hello(name: str) -> str:
    """A simple component that says hello to a given name."""
    hello_text = f'Hello, {name}!'
    print(hello_text)
    return hello_text

# Define the pipeline using the @dsl.pipeline decorator
@dsl.pipeline(
    name="hello-world-pipeline",
    description="A basic pipeline that prints a greeting."
)
def hello_pipeline(recipient: str = "World") -> str:  # Add a default value
    """This pipeline runs the say_hello component."""
    hello_task = say_hello(name=recipient)
    # Run container as non-root user (UID 1000) to satisfy runAsNonRoot enforcement
    kubernetes.set_security_context(
        hello_task,
        run_as_user=1000,
        run_as_non_root=True,
    )
    return hello_task.output  # Return the output of the component


if __name__ == "__main__":
    # Compile the pipeline into a YAML file
    compiler.Compiler().compile(hello_pipeline, 'hello_world_pipeline.yaml')

