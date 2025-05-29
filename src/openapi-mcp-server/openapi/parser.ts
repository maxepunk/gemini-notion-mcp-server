import type { OpenAPIV3, OpenAPIV3_1 } from 'openapi-types';
import type { JSONSchema7 as IJsonSchema } from 'json-schema';
import type { ChatCompletionTool } from 'openai/resources/chat/completions';
import type { Tool } from '@anthropic-ai/sdk/resources/messages/messages';

type NewToolMethod = {
  name: string;
  description: string;
  inputSchema: IJsonSchema & { type: 'OBJECT' | 'object' }; // Allow both for internal flexibility before final output
  returnSchema?: IJsonSchema;
};

type FunctionParameters = {
  type: 'object' | 'OBJECT'; // For Gemini, this should be 'OBJECT'
  properties?: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
};

// Helper for Gemini type conversion
function toGeminiType(type?: string): string | undefined {
  if (!type) return undefined;
  const typeMapping: { [key: string]: string } = {
    'number': 'NUMBER',
    'integer': 'INTEGER', // Gemini typically uses INTEGER for integers
    'boolean': 'BOOLEAN',
    'array': 'ARRAY',
    'object': 'OBJECT',
    'string': 'STRING',
  };
  return typeMapping[type.toLowerCase()] || type.toUpperCase();
}


export class OpenAPIToMCPConverter {
  private schemaCache: Record<string, IJsonSchema> = {};
  private nameCounter: number = 0;

  constructor(private openApiSpec: OpenAPIV3.Document | OpenAPIV3_1.Document) {}

  private internalResolveRef(ref: string, resolvedRefs: Set<string>): OpenAPIV3.SchemaObject | null {
    if (!ref.startsWith('#/')) {
      return null;
    }
    if (resolvedRefs.has(ref)) {
      // console.warn(`Cyclic reference detected and avoided for: ${ref}`);
      return null;
    }

    const parts = ref.replace(/^#\//, '').split('/');
    let current: any = this.openApiSpec;
    for (const part of parts) {
      current = current[part];
      if (!current) return null;
    }
    resolvedRefs.add(ref);
    return current as OpenAPIV3.SchemaObject;
  }

  convertOpenApiSchemaToJsonSchema(
    schema: OpenAPIV3.SchemaObject | OpenAPIV3.ReferenceObject,
    resolvedRefs: Set<string>, // Pass this along to track cycles for the current conversion path
    resolveRefs: boolean = false,
    dialect?: 'gemini',
  ): IJsonSchema {
    if ('$ref' in schema) {
      const ref = schema.$ref;

      // Handle potential cycles: if this ref is already in the current resolution path, return a basic ref
      if (resolvedRefs.has(ref)) {
        // console.warn(`Cyclic $ref encountered: ${ref}. Returning a placeholder $ref to avoid infinite loop.`);
        return {
            $ref: ref.startsWith('#/components/schemas/') ? ref.replace(/^#\/components\/schemas\//, '#/$defs/') : ref,
            description: ('description' in schema ? schema.description as string : undefined) ?? `Cyclic reference to ${ref}`
        };
      }


      if (!resolveRefs) {
        if (ref.startsWith('#/components/schemas/')) {
          return {
            $ref: ref.replace(/^#\/components\/schemas\//, '#/$defs/'),
            ...('description' in schema ? { description: schema.description as string } : {}),
          };
        }
        // console.error(`Attempting to resolve ref ${ref} not found in components collection or not a schema ref.`);
      }
      
      const refSchema: IJsonSchema = { $ref: ref.startsWith('#/components/schemas/') ? ref.replace(/^#\/components\/schemas\//, '#/$defs/') : ref };
      if ('description' in schema && schema.description) {
        refSchema.description = schema.description as string;
      }

      // Cache check: only use cache if not resolving fully and no specific dialect that might alter structure.
      // For simplicity, let's assume the cache is primarily for non-dialect or non-fully-resolved refs.
      // A more sophisticated cache would key by (ref, dialect, resolveRefs).
      if (this.schemaCache[ref] && dialect !== 'gemini' && !resolveRefs) {
        return {...this.schemaCache[ref], ...(refSchema.description && {description: refSchema.description})};
      }
      
      resolvedRefs.add(ref); // Add to current path before resolving
      const resolvedOpenApiSchema = this.internalResolveRef(ref, new Set()); // internalResolveRef uses its own Set for its direct path
      resolvedRefs.delete(ref); // Remove from current path after resolving

      if (!resolvedOpenApiSchema) {
        // console.error(`Failed to resolve ref ${ref}`);
        return {
          $ref: ref.startsWith('#/components/schemas/') ? ref.replace(/^#\/components\/schemas\//, '#/$defs/') : ref,
          description: ('description' in schema ? ((schema.description as string) ?? `Unresolved reference: ${ref}`) : `Unresolved reference: ${ref}`),
        };
      } else {
        // Pass a *new* Set for resolvedRefs down the recursive call to avoid interference between sibling $refs.
        const converted = this.convertOpenApiSchemaToJsonSchema(resolvedOpenApiSchema, new Set(resolvedRefs), resolveRefs, dialect);
        if (dialect !== 'gemini' && !resolveRefs) { // Only cache "standard" conversions for now
             this.schemaCache[ref] = converted;
        }
        // Ensure the original $ref's description isn't lost if the resolved one doesn't have one
        if (refSchema.description && !converted.description) {
            converted.description = refSchema.description;
        }
        return converted;
      }
    }

    const result: IJsonSchema = {};

    if (schema.type) {
      result.type = dialect === 'gemini' ? toGeminiType(schema.type as string) as IJsonSchema['type'] : schema.type as IJsonSchema['type'];
    }

    if (schema.format === 'binary') {
      result.format = 'uri-reference'; // Per original code, or 'byte' for base64 encoded
      const binaryDesc = dialect === 'gemini' ? 'Content of the file as a string (e.g. base64 encoded), or a URI reference for large files.' : 'absolute paths to local files';
      result.description = schema.description ? `${schema.description} (${binaryDesc})` : binaryDesc;
      if (dialect === 'gemini') { // Gemini expects a type, and for binary it should be string if it's to be base64
          result.type = 'STRING';
          // if it's a URI, format could be 'uri' or 'uri-reference'
      }
    } else {
      if (schema.format) {
        result.format = schema.format;
      }
      if (schema.description) {
        result.description = schema.description;
      }
    }

    if (schema.enum) {
      result.enum = schema.enum;
    }

    if (schema.default !== undefined) {
      result.default = schema.default;
    }

    if (schema.type === 'object' || (dialect === 'gemini' && result.type === 'OBJECT')) {
      result.type = dialect === 'gemini' ? 'OBJECT' : 'object';
      if (schema.properties) {
        result.properties = {};
        for (const [name, propSchema] of Object.entries(schema.properties)) {
          result.properties[name] = this.convertOpenApiSchemaToJsonSchema(propSchema, new Set(resolvedRefs), resolveRefs, dialect);
        }
      }
      if (schema.required) {
        result.required = schema.required;
      }
      if (schema.additionalProperties === true || schema.additionalProperties === undefined) {
        result.additionalProperties = true;
      } else if (schema.additionalProperties && typeof schema.additionalProperties === 'object') {
        result.additionalProperties = this.convertOpenApiSchemaToJsonSchema(schema.additionalProperties, new Set(resolvedRefs), resolveRefs, dialect);
      } else {
        result.additionalProperties = false;
      }
    }

    if (schema.type === 'array' || (dialect === 'gemini' && result.type === 'ARRAY')) {
      result.type = dialect === 'gemini' ? 'ARRAY' : 'array';
      if (schema.items) {
        result.items = this.convertOpenApiSchemaToJsonSchema(schema.items, new Set(resolvedRefs), resolveRefs, dialect);
      }
    }

    if (schema.oneOf) {
      result.oneOf = schema.oneOf.map((s) => this.convertOpenApiSchemaToJsonSchema(s, new Set(resolvedRefs), resolveRefs, dialect));
    }
    if (schema.anyOf) {
      result.anyOf = schema.anyOf.map((s) => this.convertOpenApiSchemaToJsonSchema(s, new Set(resolvedRefs), resolveRefs, dialect));
    }
    if (schema.allOf) {
      result.allOf = schema.allOf.map((s) => this.convertOpenApiSchemaToJsonSchema(s, new Set(resolvedRefs), resolveRefs, dialect));
    }

    // Preserve top-level schema description if not already set by format-specific logic
    if (schema.description && !result.description) {
        result.description = schema.description;
    }


    return result;
  }

  convertToMCPTools(): {
    tools: Record<string, { methods: NewToolMethod[] }>;
    openApiLookup: Record<string, OpenAPIV3.OperationObject & { method: string; path: string }>;
    zip: Record<string, { openApi: OpenAPIV3.OperationObject & { method: string; path: string }; mcp: NewToolMethod }>;
  } {
    const apiName = 'API';
    const openApiLookup: Record<string, OpenAPIV3.OperationObject & { method: string; path: string }> = {};
    const tools: Record<string, { methods: NewToolMethod[] }> = {
      [apiName]: { methods: [] },
    };
    const zip: Record<string, { openApi: OpenAPIV3.OperationObject & { method: string; path: string }; mcp: NewToolMethod }> = {};
    this.schemaCache = {};

    for (const [path, pathItem] of Object.entries(this.openApiSpec.paths || {})) {
      if (!pathItem) continue;

      for (const [method, operation] of Object.entries(pathItem)) {
        if (!this.isOperation(method, operation)) continue;
        
        const mcpMethod = this.convertOperationToMCPMethod(operation, method, path, 'gemini');
        if (mcpMethod) {
          const uniqueName = this.ensureUniqueName(mcpMethod.name);
          mcpMethod.name = uniqueName;
          tools[apiName]!.methods.push(mcpMethod);
          openApiLookup[apiName + '-' + uniqueName] = { ...operation, method, path };
          zip[apiName + '-' + uniqueName] = { openApi: { ...operation, method, path }, mcp: mcpMethod };
        }
      }
    }
    return { tools, openApiLookup, zip };
  }

  convertToOpenAITools(): ChatCompletionTool[] {
    const tools: ChatCompletionTool[] = [];
    this.schemaCache = {};
    for (const [path, pathItem] of Object.entries(this.openApiSpec.paths || {})) {
      if (!pathItem) continue;
      for (const [method, operation] of Object.entries(pathItem)) {
        if (!this.isOperation(method, operation)) continue;
        const parameters = this.convertOperationToJsonSchemaInternal(operation, method, path); // No dialect
        const tool: ChatCompletionTool = {
          type: 'function',
          function: {
            name: operation.operationId!,
            description: operation.summary || operation.description || '',
            parameters: parameters as unknown as FunctionParameters,
          },
        };
        tools.push(tool);
      }
    }
    return tools;
  }

  convertToAnthropicTools(): Tool[] {
    const tools: Tool[] = [];
    this.schemaCache = {};
    for (const [path, pathItem] of Object.entries(this.openApiSpec.paths || {})) {
      if (!pathItem) continue;
      for (const [method, operation] of Object.entries(pathItem)) {
        if (!this.isOperation(method, operation)) continue;
        const parameters = this.convertOperationToJsonSchemaInternal(operation, method, path); // No dialect
        const tool: Tool = {
          name: operation.operationId!,
          description: operation.summary || operation.description || '',
          input_schema: parameters as Tool['input_schema'],
        };
        tools.push(tool);
      }
    }
    return tools;
  }

  private convertComponentsToJsonSchema(dialect?: 'gemini'): Record<string, IJsonSchema> {
    const components = this.openApiSpec.components || {};
    const schema: Record<string, IJsonSchema> = {};
    for (const [key, value] of Object.entries(components.schemas || {})) {
      // For $defs, we want to fully resolve and convert them according to the dialect.
      // The 'resolveRefs = true' ensures we get the full definition.
      // The 'new Set()' for resolvedRefs gives a fresh cycle detection context for each component.
      schema[key] = this.convertOpenApiSchemaToJsonSchema(value, new Set(), true, dialect);
    }
    return schema;
  }
  
  private convertOperationToJsonSchemaInternal(
    operation: OpenAPIV3.OperationObject,
    method: string,
    path: string,
    dialect?: 'gemini'
  ): IJsonSchema & { type: 'object' | 'OBJECT' } {
    const schemaType = dialect === 'gemini' ? 'OBJECT' : 'object';
    // Pass 'true' for resolveRefs to convertComponentsToJsonSchema to ensure $defs are fully resolved.
    const schema: IJsonSchema & { type: 'object' | 'OBJECT' } = {
      type: schemaType,
      properties: {},
      required: [],
      // $defs should contain fully resolved and dialect-converted component schemas.
      $defs: this.convertComponentsToJsonSchema(dialect),
    };

    if (operation.parameters) {
      for (const param of operation.parameters) {
        const paramObj = this.resolveParameter(param);
        if (paramObj && paramObj.schema) {
          // For parameters, we don't want to fully resolve internal $refs within their schemas at this stage,
          // let them be handled by the final schema consumer if needed.
          // Pass 'false' for resolveRefs here.
          const paramSchema = this.convertOpenApiSchemaToJsonSchema(paramObj.schema, new Set(), false, dialect);
          if (paramObj.description) {
            paramSchema.description = paramObj.description;
          }
          schema.properties![paramObj.name] = paramSchema;
          if (paramObj.required) {
            schema.required!.push(paramObj.name);
          }
        }
      }
    }

    if (operation.requestBody) {
      const bodyObj = this.resolveRequestBody(operation.requestBody);
      if (bodyObj?.content) {
        const contentType = Object.keys(bodyObj.content)[0];
        if (bodyObj.content[contentType]?.schema) {
            const bodyContentSchema = bodyObj.content[contentType].schema;
            // For requestBody schema, also don't fully resolve internal $refs yet. Pass 'false'.
            const convertedBodySchema = this.convertOpenApiSchemaToJsonSchema(bodyContentSchema, new Set(), false, dialect);

            if (contentType === 'application/json') {
                 if ((convertedBodySchema.type === 'object' || convertedBodySchema.type === 'OBJECT') && convertedBodySchema.properties) {
                    for (const [name, propSchema] of Object.entries(convertedBodySchema.properties)) {
                        schema.properties![name] = propSchema;
                    }
                    if (convertedBodySchema.required) {
                        schema.required!.push(...convertedBodySchema.required);
                    }
                } else {
                    schema.properties!['body'] = convertedBodySchema;
                    if (bodyObj.required) {
                         schema.required!.push('body');
                    }
                }
            } else if (contentType === 'multipart/form-data') {
                 if ((convertedBodySchema.type === 'object' || convertedBodySchema.type === 'OBJECT') && convertedBodySchema.properties) {
                    for (const [name, propSchema] of Object.entries(convertedBodySchema.properties)) {
                        schema.properties![name] = propSchema;
                    }
                    if (convertedBodySchema.required) {
                        schema.required!.push(...convertedBodySchema.required!);
                    }
                }
            }
        }
      }
    }
    return schema;
  }

  private isOperation(method: string, operation: any): operation is OpenAPIV3.OperationObject {
    return ['get', 'post', 'put', 'delete', 'patch'].includes(method.toLowerCase());
  }

  private isParameterObject(param: OpenAPIV3.ParameterObject | OpenAPIV3.ReferenceObject): param is OpenAPIV3.ParameterObject {
    return !('$ref' in param);
  }

  private isRequestBodyObject(body: OpenAPIV3.RequestBodyObject | OpenAPIV3.ReferenceObject): body is OpenAPIV3.RequestBodyObject {
    return !('$ref' in body);
  }

  private resolveParameter(param: OpenAPIV3.ParameterObject | OpenAPIV3.ReferenceObject): OpenAPIV3.ParameterObject | null {
    if (this.isParameterObject(param)) {
      return param;
    } else {
      const resolved = this.internalResolveRef(param.$ref, new Set()); // Use new Set for independent resolution path
      if (resolved && (resolved as OpenAPIV3.ParameterObject).name !== undefined) {
        return resolved as OpenAPIV3.ParameterObject;
      }
    }
    return null;
  }

  private resolveRequestBody(body: OpenAPIV3.RequestBodyObject | OpenAPIV3.ReferenceObject): OpenAPIV3.RequestBodyObject | null {
    if (this.isRequestBodyObject(body)) {
      return body;
    } else {
      const resolved = this.internalResolveRef(body.$ref, new Set()); // Use new Set
      if (resolved) {
        return resolved as OpenAPIV3.RequestBodyObject;
      }
    }
    return null;
  }

  private resolveResponse(response: OpenAPIV3.ResponseObject | OpenAPIV3.ReferenceObject): OpenAPIV3.ResponseObject | null {
    if ('$ref' in response) {
      const resolved = this.internalResolveRef(response.$ref, new Set()); // Use new Set
      return resolved ? resolved as OpenAPIV3.ResponseObject : null;
    }
    return response;
  }

  private convertOperationToMCPMethod(operation: OpenAPIV3.OperationObject, method: string, path: string, dialect?: 'gemini'): NewToolMethod | null {
    if (!operation.operationId) {
      console.warn(`Operation without operationId at ${method} ${path}`);
      return null;
    }
    const methodName = operation.operationId;
    const inputSchema = this.convertOperationToJsonSchemaInternal(operation, method, path, dialect);

    let description = operation.summary || operation.description || '';
    if (operation.responses) {
      const errorResponses = Object.entries(operation.responses)
        .filter(([code]) => code.startsWith('4') || code.startsWith('5'))
        .map(([code, resp]) => {
          const responseObj = this.resolveResponse(resp);
          let errorDesc = responseObj?.description || '';
          return `${code}: ${errorDesc}`;
        });
      if (errorResponses.length > 0) {
        description += '\\nError Responses:\\n' + errorResponses.join('\\n');
      }
    }
    const returnSchema = this.extractResponseType(operation.responses, dialect);

    return {
      name: methodName,
      description,
      inputSchema: inputSchema as IJsonSchema & { type: 'OBJECT' | 'object' },
      ...(returnSchema ? { returnSchema } : {}),
    };
  }

  private extractResponseType(responses: OpenAPIV3.ResponsesObject | undefined, dialect?: 'gemini'): IJsonSchema | null {
    const successResponse = responses?.['200'] || responses?.['201'] || responses?.['202'] || responses?.['204'];
    if (!successResponse) return null;

    const responseObj = this.resolveResponse(successResponse);
    if (!responseObj || !responseObj.content) return null;

    if (responseObj.content['application/json']?.schema) {
      const returnSchema = this.convertOpenApiSchemaToJsonSchema(responseObj.content['application/json'].schema, new Set(), false, dialect);
      returnSchema['$defs'] = this.convertComponentsToJsonSchema(dialect); // Ensure $defs are also dialect-converted
      if (responseObj.description && !returnSchema.description) {
        returnSchema.description = responseObj.description;
      }
      return returnSchema;
    }
    if (responseObj.content['image/png'] || responseObj.content['image/jpeg'] || responseObj.content['application/octet-stream']) {
      return { type: dialect === 'gemini' ? 'STRING' : 'string', format: 'binary', description: responseObj.description || 'Binary file content' };
    }
    // Fallback for other content types
    const firstContentType = Object.keys(responseObj.content)[0];
    if(firstContentType && responseObj.content[firstContentType]?.schema) {
        const returnSchema = this.convertOpenApiSchemaToJsonSchema(responseObj.content[firstContentType].schema!, new Set(), false, dialect);
        returnSchema['$defs'] = this.convertComponentsToJsonSchema(dialect);
        if (responseObj.description && !returnSchema.description) {
            returnSchema.description = responseObj.description;
        }
        return returnSchema;
    }

    return { type: dialect === 'gemini' ? 'STRING' : 'string', description: responseObj.description || 'Plain text response' };
  }

  private ensureUniqueName(name: string): string {
    if (name.length <= 64) {
      return name;
    }
    const truncatedName = name.slice(0, 64 - 5);
    const uniqueSuffix = this.generateUniqueSuffix();
    return `${truncatedName}-${uniqueSuffix}`;
  }

  private generateUniqueSuffix(): string {
    this.nameCounter += 1;
    return this.nameCounter.toString().padStart(4, '0');
  }
}
