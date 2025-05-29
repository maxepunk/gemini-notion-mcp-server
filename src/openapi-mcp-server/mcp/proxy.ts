import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, ListToolsRequestSchema, Tool } from '@modelcontextprotocol/sdk/types.js';
import { JSONSchema7 as IJsonSchema } from 'json-schema';
import { OpenAPIToMCPConverter } from '../openapi/parser'; // Your modified parser
import { HttpClient, HttpClientError } from '../client/http-client';
import { OpenAPIV3 } from 'openapi-types';
import { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';

// MCP Tool's inputSchema expects specific JSONSchema7 types.
// The root type should be 'object'. Properties within will be Gemini-formatted by the parser.
type MCPToolInputSchema = IJsonSchema & { type: 'object' };

// This type comes from the parser, where inputSchema.type could be 'OBJECT' (Gemini) or 'object'
type ParserToolMethod = {
  name: string;
  description: string;
  inputSchema: IJsonSchema & { type: 'OBJECT' | 'object' };
  returnSchema?: IJsonSchema;
};

type ParserToolsOutput = {
    methods: Array<ParserToolMethod>;
};

export class MCPProxy {
  private server: Server;
  private httpClient: HttpClient;
  // Stores tools as provided by the parser (parser's 'gemini' dialect output)
  private rawMcTools: Record<string, ParserToolsOutput>;
  // Lookup for OpenAPI operation details, keyed by an internal ID (e.g., 'API-' + uniqueName)
  private openApiLookup: Record<string, OpenAPIV3.OperationObject & { method: string; path: string }>;
  // Map from the name exposed to the LLM/MCP client back to the internal openApiLookup key
  private exposedNameToInternalKeyMap: Map<string, string>;


  constructor(name: string, openApiSpec: OpenAPIV3.Document) {
    this.server = new Server({ name, version: '1.0.0' }, { capabilities: { tools: {} } });
    const baseUrl = openApiSpec.servers?.[0].url;
    if (!baseUrl) {
      throw new Error('No base URL found in OpenAPI spec');
    }
    this.httpClient = new HttpClient(
      {
        baseUrl,
        headers: this.parseHeadersFromEnv(),
      },
      openApiSpec,
    );

    const converter = new OpenAPIToMCPConverter(openApiSpec);
    // convertToMCPTools() in the modified parser now applies the 'gemini' dialect internally.
    const { tools: parsedTools, openApiLookup: parsedOpenApiLookup } = converter.convertToMCPTools();
    
    this.rawMcTools = parsedTools as Record<string, ParserToolsOutput>; // Cast if NewToolMethod changed
    this.openApiLookup = parsedOpenApiLookup;
    this.exposedNameToInternalKeyMap = new Map();

    this.setupHandlers();
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const mcpToolsForClient: Tool[] = [];
      this.exposedNameToInternalKeyMap.clear(); // Rebuild map each time tools are listed

      Object.entries(this.rawMcTools).forEach(([toolGroupName, def]) => {
        def.methods.forEach(parserMethod => {
          // parserMethod.name is the uniqueName from the parser (e.g., operationId, possibly with suffix)
          // The internal key for openApiLookup is 'API-' + parserMethod.name
          const internalLookupKey = `${toolGroupName}-${parserMethod.name}`;

          // This is the name that will be shown to the LLM and used in CallTool requests.
          // It must be consistent and adhere to any LLM limitations (e.g., length via truncateToolName).
          const exposedToolName = this.truncateToolName(internalLookupKey);
          
          // Store the mapping from the (potentially truncated) exposed name to the full internal key
          this.exposedNameToInternalKeyMap.set(exposedToolName, internalLookupKey);

          const mcpCompatibleInputSchema: MCPToolInputSchema = {
            ...(parserMethod.inputSchema as IJsonSchema), // Properties within are already Gemini-formatted
            type: 'object', // MCP SDK/client expects 'object' at the root of inputSchema
          };
          // If parser set root to 'OBJECT' for Gemini, ensure it's 'object' for the MCP Tool object.
          if (parserMethod.inputSchema.type === 'OBJECT') {
             mcpCompatibleInputSchema.type = 'object';
          }

          mcpToolsForClient.push({
            name: exposedToolName,
            description: parserMethod.description,
            inputSchema: mcpCompatibleInputSchema,
          });
        });
      });

      return { tools: mcpToolsForClient };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name: requestedToolName, arguments: params } = request.params;

      // Use the map to find the full internal key
      const internalLookupKey = this.exposedNameToInternalKeyMap.get(requestedToolName);

      if (!internalLookupKey) {
        console.error(`Tool name "${requestedToolName}" not found in exposedNameToInternalKeyMap. Available exposed names: [${Array.from(this.exposedNameToInternalKeyMap.keys()).join(', ')}]`);
        throw new Error(`Method ${requestedToolName} not found or mapping failed.`);
      }

      const operation = this.openApiLookup[internalLookupKey];
      
      if (!operation) {
         // This should ideally not happen if the map is correct
        console.error(`Internal lookup key "${internalLookupKey}" (from exposed name "${requestedToolName}") not found in openApiLookup. Available internal keys: [${Object.keys(this.openApiLookup).join(', ')}]`);
        throw new Error(`Internal error: Operation for ${requestedToolName} (key: ${internalLookupKey}) not found.`);
      }

      try {
        const response = await this.httpClient.executeOperation(operation, params);
        return {
          content: [
            {
              type: 'text', // MCP primarily uses text for results
              text: JSON.stringify(response.data),
            },
          ],
        };
      } catch (error) {
        console.error(`Error executing tool ${requestedToolName} (internal key: ${internalLookupKey}):`, error);
        if (error instanceof HttpClientError) {
          const errorData = error.data?.response?.data ?? error.data ?? {};
          return {
            content: [
              {
                type: 'text',
                text: JSON.stringify({
                  status: 'error',
                  message: error.message,
                  details: (typeof errorData === 'object' ? errorData : { data: errorData }),
                  statusCode: error.data?.response?.status
                }),
              },
            ],
          };
        }
        // For other errors, rethrow to let the MCP SDK handle generic error formatting if it does.
        // Or, provide a structured error as well.
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify({
                        status: 'error',
                        message: (error instanceof Error ? error.message : 'An unknown error occurred'),
                    }),
                }
            ]
        }
      }
    });
  }

  private parseHeadersFromEnv(): Record<string, string> {
    const headersJson = process.env.OPENAPI_MCP_HEADERS;
    if (!headersJson) {
      return {};
    }
    try {
      const headers = JSON.parse(headersJson);
      if (typeof headers !== 'object' || headers === null) {
        console.warn('OPENAPI_MCP_HEADERS environment variable must be a JSON object, got:', typeof headers);
        return {};
      }
      return headers;
    } catch (error) {
      console.warn('Failed to parse OPENAPI_MCP_HEADERS environment variable:', error);
      return {};
    }
  }

  private truncateToolName(name: string): string {
    const maxLength = 64; 
    if (name.length <= maxLength) {
      return name;
    }
    // Simple truncation. Consider if a more sophisticated approach is needed to avoid collisions
    // if multiple long names truncate to the same short name (though ensureUniqueName in parser should help).
    console.warn(`Tool name "${name}" was truncated to "${name.slice(0, maxLength)}" due to length limits.`);
    return name.slice(0, maxLength);
  }

  async connect(transport: Transport) {
    await this.server.connect(transport);
  }

  getServer() {
    return this.server;
  }
}
